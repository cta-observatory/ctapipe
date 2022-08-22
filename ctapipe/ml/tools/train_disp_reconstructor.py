import astropy.units as u
import numpy as np

from ctapipe.core import Tool
from ctapipe.core.traits import Bool, Int, Path, Unicode
from ctapipe.io import EventSource, TableLoader

from ..apply import CrossValidator, DispClassifier, DispRegressor
from ..coordinates import horizontal_to_telescope
from ..preprocessing import check_valid_rows
from ..sklearn import Classifier, Regressor


def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


class TrainDispReconstructor(Tool):
    """Train two ML models for origin reconstruction using the disp method"""

    name = "ctapipe-train-disp"
    description = __doc__

    output_path_reg = Path(
        default_value=None, allow_none=False, directory_ok=False
    ).tag(config=True)

    output_path_clf = Path(
        default_value=None, allow_none=False, directory_ok=False
    ).tag(config=True)

    n_events = Int(default_value=None, allow_none=True).tag(config=True)
    random_seed = Int(default_value=0).tag(config=True)
    project_disp = Bool(default_value=False).tag(config=True)

    true_alt_column = Unicode(default_value=None, allow_none=False).tag(config=True)
    true_az_column = Unicode(default_value=None, allow_none=False).tag(config=True)
    delta_column = Unicode(default_value=None, allow_none=False).tag(config=True)
    cog_x_column = Unicode(default_value=None, allow_none=False).tag(config=True)
    cog_y_column = Unicode(default_value=None, allow_none=False).tag(config=True)

    aliases = {
        "input": "TableLoader.input_url",
        "output_regressor": "TrainDispReconstructor.output_path_reg",
        "output_classifier": "TrainDispReconstructor.output_path_clf",
    }

    classes = [TableLoader, Regressor, Classifier, CrossValidator]

    def setup(self):
        """"""
        self.loader = TableLoader(
            parent=self,
            load_dl1_images=False,
            load_dl1_parameters=True,
            load_dl2=True,
            load_simulated=True,
            load_instrument=True,
        )
        # Atm pointing information can only be accessed using EventSource
        # Pointing information will be added to HDF5 tables soon, see #1902
        self.source = EventSource(self.loader.input_url, max_events=1)

        self.regressor = DispRegressor(self.loader.subarray, parent=self)
        self.classifier = DispClassifier(self.loader.subarray, parent=self)
        self.cross_validate_reg = CrossValidator(
            parent=self, model_component=self.regressor
        )
        self.cross_validate_clf = CrossValidator(
            parent=self, model_component=self.classifier
        )
        self.rng = np.random.default_rng(self.random_seed)

    def start(self):
        """"""
        types = self.loader.subarray.telescope_types
        self.log.info("Inputfile: %s", self.loader.input_url)

        for event in self.source:
            pointing_alt = event.pointing.array_altitude.to(u.deg)
            pointing_az = event.pointing.array_azimuth.to(u.deg)

        self.log.info("Simulated pointing altitude: %s", pointing_alt)
        self.log.info("Simulated pointing azimuth: %s", pointing_az)

        self.log.info("Training models for %d types", len(types))
        for tel_type in types:
            self.log.info("Loading events for %s", tel_type)
            table_reg, table_clf = self._read_table(tel_type, pointing_alt, pointing_az)

            self.log.info("Train regressor on %s events", len(table_reg))
            self.cross_validate_reg(tel_type, table_reg)

            self.log.info("Train classifier on %s events", len(table_clf))
            self.cross_validate_clf(tel_type, table_clf)

            self.log.info("Performing final fit for %s", tel_type)
            self.regressor.model.fit(tel_type, table_reg)
            self.classifier.model.fit(tel_type, table_clf)
            self.log.info("done")

    def _read_table(
        self,
        telescope_type,
        pointing_altitude: u.Quantity,
        pointing_azimuth: u.Quantity,
    ):
        table = self.loader.read_telescope_events([telescope_type])
        self.log.info("Events read from input: %d", len(table))

        # TODO: De-spaghettify the following

        # Allow separate quality queries/ event lists for training the two models
        mask_reg = self.regressor.qualityquery.get_table_mask(table)
        table_reg = table[mask_reg]
        self.log.info(
            "Events after applying quality query for regressor: %d", len(table_reg)
        )

        mask_clf = self.classifier.qualityquery.get_table_mask(table)
        table_clf = table[mask_clf]
        self.log.info(
            "Events after applying quality query for classifier: %d", len(table_clf)
        )

        table_reg = self.regressor.generate_features(table_reg)
        table_clf = self.classifier.generate_features(table_clf)

        true_norm, _ = self._get_true_disp(
            table_reg, pointing_altitude, pointing_azimuth
        )
        _, true_sign = self._get_true_disp(
            table_clf, pointing_altitude, pointing_azimuth
        )

        table_reg = table_reg[self.regressor.model.features]
        table_clf = table_clf[self.classifier.model.features]

        table_reg[self.regressor.target] = true_norm
        table_clf[self.classifier.target] = true_sign

        valid_reg = check_valid_rows(table_reg)
        if np.any(~valid_reg):
            self.log.warning("Dropping non-predictable events for regressor.")
            table_reg = table_reg[valid_reg]

        valid_clf = check_valid_rows(table_clf)
        if np.any(~valid_clf):
            self.log.warning("Dropping non-predictable events for classifier.")
            table_clf = table_clf[valid_clf]

        if self.n_events is not None:
            n_events = min(self.n_events, len(table_reg))
            idx = self.rng.choice(len(table_reg), n_events, replace=False)
            idx.sort()
            table_reg = table_reg[idx]

            n_events = min(self.n_events, len(table_clf))
            idx = self.rng.choice(len(table_clf), n_events, replace=False)
            idx.sort()
            table_clf = table_clf[idx]

        return table_reg, table_clf

    def _get_true_disp(
        self, table, pointing_altitude: u.Quantity, pointing_azimuth: u.Quantity
    ):
        fov_lon, fov_lat = horizontal_to_telescope(
            alt=table[self.true_alt_column],
            az=table[self.true_az_column],
            pointing_alt=pointing_altitude,
            pointing_az=pointing_azimuth,
        )
        # should all this be calculated using delta, cog_x, cog_y based on the true image?

        # numpy's trigonometric functions need radians
        delta = table[self.delta_column].to(u.rad)

        delta_x = fov_lon - table[self.cog_x_column]
        delta_y = fov_lat - table[self.cog_y_column]

        true_disp = np.cos(delta) * delta_x + np.sin(delta) * delta_y
        true_sign = np.sign(true_disp)

        if self.project_disp:
            true_norm = np.abs(true_disp)
        else:
            true_norm = euclidean_distance(
                fov_lon, fov_lat, table[self.cog_x_column], table[self.cog_y_column]
            )

        return true_norm, true_sign

    def finish(self):
        self.log.info("Writing output")
        self.regressor.write(self.output_path_reg)
        self.classifier.write(self.output_path_clf)
        # write complete cv performance in two separate files, if at least one output path is given
        if (
            self.cross_validate_reg.output_path
            and self.cross_validate_clf.output_path
            and (
                self.cross_validate_reg.output_path
                != self.cross_validate_clf.output_path
            )
        ):
            self.cross_validate_reg.write()
            self.cross_validate_clf.write()
        else:
            if (
                self.cross_validate_reg.output_path
                and not self.cross_validate_clf.output_path
            ):
                outpath = self.cross_validate_reg.output_path
            else:  # covers only clf output path given and same output path for both
                outpath = self.cross_validate_clf.output_path

            self.cross_validate_reg.output_path = outpath.parent / (
                outpath.stem + "_regressor" + outpath.suffix
            )
            self.cross_validate_clf.output_path = outpath.parent / (
                outpath.stem + "_classifier" + outpath.suffix
            )

            self.cross_validate_reg.write()
            self.cross_validate_clf.write()

        self.loader.close()


def main():
    TrainDispReconstructor().run()


if __name__ == "__main__":
    main()
