"""
Tool for training the EnergyRegressor
"""
import numpy as np

from ctapipe.core import Tool
from ctapipe.core.traits import Int, IntTelescopeParameter, Path
from ctapipe.exceptions import TooFewEvents
from ctapipe.io import TableLoader
from ctapipe.reco import CrossValidator, EnergyRegressor
from ctapipe.reco.preprocessing import check_valid_rows

__all__ = [
    "TrainEnergyRegressor",
]


class TrainEnergyRegressor(Tool):
    """
    Tool to train a `~ctapipe.reco.EnergyRegressor` on dl1b/dl2 data.

    The tool first performs a cross validation to give an initial estimate
    on the quality of the estimation and then finally trains one model
    per telescope type on the full dataset.
    """

    name = "ctapipe-train-energy-regressor"
    description = __doc__

    examples = """
    ctapipe-train-energy-regressor \\
        --config train_energy_regressor.yaml \\
        --input gamma.dl2.h5 \\
        --output energy_regressor.pkl
    """

    output_path = Path(
        default_value=None,
        allow_none=False,
        directory_ok=False,
        help=(
            "Output path for the trained reconstructor."
            " At the moment, pickle is the only supported format."
        ),
    ).tag(config=True)

    n_events = IntTelescopeParameter(
        default_value=None,
        allow_none=True,
        help=(
            "Number of events for training the model."
            " If not given, all available events will be used."
        ),
    ).tag(config=True)

    random_seed = Int(
        default_value=0, help="Random seed for sampling and cross validation"
    ).tag(config=True)

    aliases = {
        ("i", "input"): "TableLoader.input_url",
        ("o", "output"): "TrainEnergyRegressor.output_path",
        "n-events": "TrainEnergyRegressor.n_events",
        "cv-output": "CrossValidator.output_path",
    }

    classes = [
        TableLoader,
        EnergyRegressor,
        CrossValidator,
    ]

    def setup(self):
        """
        Initialize components from config
        """
        self.loader = self.enter_context(
            TableLoader(
                parent=self,
                load_dl1_images=False,
                load_dl1_parameters=True,
                load_dl2=True,
                load_simulated=True,
                load_instrument=True,
            )
        )
        self.n_events.attach_subarray(self.loader.subarray)

        self.regressor = EnergyRegressor(self.loader.subarray, parent=self)
        self.cross_validate = CrossValidator(
            parent=self, model_component=self.regressor
        )
        self.rng = np.random.default_rng(self.random_seed)
        self.check_output(self.output_path, self.cross_validate.output_path)

    def start(self):
        """
        Train models per telescope type.
        """

        types = self.loader.subarray.telescope_types
        self.log.info("Inputfile: %s", self.loader.input_url)
        self.log.info("Training models for %d types", len(types))
        for tel_type in types:
            self.log.info("Loading events for %s", tel_type)
            table = self._read_table(tel_type)

            self.log.info("Train on %s events", len(table))
            self.cross_validate(tel_type, table)

            self.log.info("Performing final fit for %s", tel_type)
            self.regressor.fit(tel_type, table)
            self.log.info("done")

    def _read_table(self, telescope_type):
        table = self.loader.read_telescope_events([telescope_type])
        self.log.info("Events read from input: %d", len(table))
        if len(table) == 0:
            raise TooFewEvents(
                f"Input file does not contain any events for telescope type {telescope_type}"
            )

        mask = self.regressor.quality_query.get_table_mask(table)
        table = table[mask]
        self.log.info("Events after applying quality query: %d", len(table))
        if len(table) == 0:
            raise TooFewEvents(
                f"No events after quality query for telescope type {telescope_type}"
            )

        table = self.regressor.feature_generator(table, subarray=self.loader.subarray)

        feature_names = self.regressor.features + [self.regressor.target]
        table = table[feature_names]

        valid = check_valid_rows(table)
        if np.any(~valid):
            self.log.warning("Dropping non-predictable events.")
            table = table[valid]

        n_events = self.n_events.tel[telescope_type]
        if n_events is not None:
            if n_events > len(table):
                self.log.warning(
                    "Number of events in table (%d) is less than requested number of events %d",
                    len(table),
                    n_events,
                )
            else:
                self.log.info("Sampling %d events", n_events)
                idx = self.rng.choice(len(table), n_events, replace=False)
                idx.sort()
                table = table[idx]

        return table

    def finish(self):
        """
        Write-out trained models and cross-validation results.
        """
        self.log.info("Writing output")
        self.regressor.write(self.output_path, overwrite=self.overwrite)
        if self.cross_validate.output_path:
            self.cross_validate.write(overwrite=self.overwrite)
        self.loader.close()


def main():
    TrainEnergyRegressor().run()


if __name__ == "__main__":
    main()
