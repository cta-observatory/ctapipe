import astropy.units as u
import numpy as np

from ctapipe.coordinates.disp import horizontal_to_telescope
from ctapipe.core import Tool
from ctapipe.core.traits import Bool, Int, Path
from ctapipe.io import EventSource, TableLoader

from ..apply import CrossValidator, DispClassifier, DispRegressor
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

        # Atm pointing information can only be accessed using EventSource
        # Pointing information will be added to HDF5 tables soon, see #1902
        for event in EventSource(self.loader.input_url, max_events=1):
            self.pointing_alt = event.pointing.array_altitude.to(u.deg)
            self.pointing_az = event.pointing.array_azimuth.to(u.deg)

        self.log.info("Simulated pointing altitude: %s", self.pointing_alt)
        self.log.info("Simulated pointing azimuth: %s", self.pointing_az)

        self.log.info("Training models for %d types", len(types))
        for tel_type in types:
            self.log.info("Loading events for %s", tel_type)
            table_reg, table_clf = self._read_table(tel_type)

            self.log.info("Train regressor on %s events", len(table_reg))
            self.cross_validate_reg(tel_type, table_reg)

            self.log.info("Train classifier on %s events", len(table_clf))
            self.cross_validate_clf(tel_type, table_clf)

            self.log.info("Performing final fit for %s", tel_type)
            self.regressor.model.fit(tel_type, table_reg)
            self.classifier.model.fit(tel_type, table_clf)
            self.log.info("done")

    def _read_table(self, telescope_type):
        table = self.loader.read_telescope_events([telescope_type])
        self.log.info("Events read from input: %d", len(table))

        # Allow separate quality queries/ event lists for training the two models
        # but dont load the events two times to shorten runtime
        table_reg = self._get_reconstructor_table(table, self.regressor)
        table_clf = self._get_reconstructor_table(table, self.classifier)

        return table_reg, table_clf

    def _get_reconstructor_table(self, table, reconstructor):
        mask = reconstructor.qualityquery.get_table_mask(table)
        table = table[mask]
        self.log.info(
            "Events after applying quality query for %s: %d",
            reconstructor.model.model_cls,
            len(table),
        )

        table = reconstructor.generate_features(table)

        if reconstructor == self.regressor:
            target_values, _ = self._get_true_disp(table)
        else:
            _, target_values = self._get_true_disp(table)

        table = table[reconstructor.model.features]

        table[reconstructor.target] = target_values

        valid = check_valid_rows(table)
        if np.any(~valid):
            self.log.warning(
                "Dropping non-predictable events for %s.", reconstructor.model.model_cls
            )
            table = table[valid]

        if self.n_events is not None:
            n_events = min(self.n_events, len(table))
            idx = self.rng.choice(len(table), n_events, replace=False)
            idx.sort()
            table = table[idx]

        return table

    def _get_true_disp(self, table):
        fov_lon, fov_lat = horizontal_to_telescope(
            alt=table["true_alt"],
            az=table["true_az"],
            pointing_alt=self.pointing_alt,
            pointing_az=self.pointing_az,
        )
        # should all this be calculated using delta, cog_x, cog_y based on the true image?

        # numpy's trigonometric functions need radians
        delta = table["hillas_psi"].to(u.rad)

        delta_x = fov_lon - table["hillas_fov_lon"]
        delta_y = fov_lat - table["hillas_fov_lat"]

        true_disp = np.cos(delta) * delta_x + np.sin(delta) * delta_y
        true_sign = np.sign(true_disp)

        if self.project_disp:
            true_norm = np.abs(true_disp)
        else:
            true_norm = euclidean_distance(
                fov_lon, fov_lat, table["hillas_fov_lon"], table["hillas_fov_lat"]
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
        elif self.cross_validate_reg.output_path or self.cross_validate_clf.output_path:
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
