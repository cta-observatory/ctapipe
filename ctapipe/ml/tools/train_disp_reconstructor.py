import astropy.units as u
import numpy as np

from ctapipe.coordinates.disp import horizontal_to_telescope
from ctapipe.core import Tool
from ctapipe.core.traits import Bool, Int, Path
from ctapipe.io import EventSource, TableLoader

from ..apply import CrossValidator, DispReconstructor
from ..preprocessing import check_valid_rows
from ..sklearn import Classifier, Regressor


class TrainDispReconstructor(Tool):
    """Train two ML models for origin reconstruction using the disp method"""

    name = "ctapipe-train-disp"
    description = __doc__

    output_path = Path(default_value=None, allow_none=False, directory_ok=False).tag(
        config=True
    )

    n_events = Int(default_value=None, allow_none=True).tag(config=True)
    random_seed = Int(default_value=0).tag(config=True)
    project_disp = Bool(
        default_value=False,
        help="Project true source position on main shower axis for true |disp| calculation",
    ).tag(config=True)

    aliases = {
        "input": "TableLoader.input_url",
        "output": "TrainDispReconstructor.output_path",
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

        self.models = DispReconstructor(self.loader.subarray, parent=self)
        self.cross_validate = CrossValidator(parent=self, model_component=self.models)
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
            table = self._read_table(tel_type)

            self.log.info("Train models on %s events", len(table))
            self.cross_validate(tel_type, table)

            self.log.info("Performing final fit for %s", tel_type)
            self.models.norm_regressor.fit(tel_type, table)
            self.models.sign_classifier.fit(tel_type, table)
            self.log.info("done")

    def _read_table(self, telescope_type):
        table = self.loader.read_telescope_events([telescope_type])
        self.log.info("Events read from input: %d", len(table))

        mask = self.models.qualityquery.get_table_mask(table)
        table = table[mask]
        self.log.info("Events after applying quality query: %d", len(table))

        table = self.models.generate_features(table)

        true_norm, true_sign = self._get_true_disp(table)

        # get a list of all features used for norm AND sign
        feature_names_combined = self.models.norm_regressor.features
        for feature in self.models.sign_classifier.features:
            if feature not in feature_names_combined:
                feature_names_combined.append(feature)

        table = table[feature_names_combined]
        table[self.models.target_norm] = true_norm
        table[self.models.target_sign] = true_sign

        valid = check_valid_rows(table)
        if np.any(~valid):
            self.log.warning("Dropping non-predicable events.")
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

        # numpy's trigonometric functions need radians
        psi = table["hillas_psi"].quantity.to_value(u.rad)
        cog_lon = table["hillas_fov_lon"].quantity
        cog_lat = table["hillas_fov_lat"].quantity

        delta_lon = fov_lon - cog_lon
        delta_lat = fov_lat - cog_lat

        true_disp = np.cos(psi) * delta_lon + np.sin(psi) * delta_lat
        true_sign = np.sign(true_disp)

        if self.project_disp:
            true_norm = np.abs(true_disp)
        else:
            true_norm = np.sqrt((fov_lon - cog_lon) ** 2 + (fov_lat - cog_lat) ** 2)

        return true_norm, true_sign.astype(np.int8)

    def finish(self):
        self.log.info("Writing output")
        self.models.write(self.output_path)
        if self.cross_validate.output_path:
            self.cross_validate.write()
        self.loader.close()


def main():
    TrainDispReconstructor().run()


if __name__ == "__main__":
    main()
