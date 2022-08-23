import numpy as np
from astropy.table import vstack

from ctapipe.core.tool import Tool
from ctapipe.core.traits import Int, Path
from ctapipe.io import TableLoader

from ..preprocessing import check_valid_rows
from ..sklearn import CrossValidator, ParticleIdClassifier


class TrainParticleIdClassifier(Tool):
    """
    Tool to train a `~ctapipe.ml.ParticleIdClassifier` on dl2 data.

    The tool first performs a cross validation to give an initial estimate
    on the quality of the estimation and then finally trains one model
    per telescope type on the full dataset.
    """

    name = "ctapipe-train-classifier"
    description = __doc__

    input_url_signal = Path(
        default_value=None,
        allow_none=False,
        directory_ok=False,
    ).tag(config=True)

    input_url_background = Path(
        default_value=None,
        allow_none=False,
        directory_ok=False,
    ).tag(config=True)

    output_path = Path(
        default_value=None,
        allow_none=False,
        directory_ok=False,
    ).tag(config=True)

    n_signal = Int(default_value=None, allow_none=True).tag(config=True)
    n_background = Int(default_value=None, allow_none=True).tag(config=True)
    random_seed = Int(default_value=0).tag(config=True)

    aliases = {
        ("s", "signal"): "TrainParticleIdClassifier.input_url_signal",
        ("b", "background"): "TrainParticleIdClassifier.input_url_background",
        ("o", "output"): "TrainParticleIdClassifier.output_path",
        "cv-output": "CrossValidator.output_path",
    }

    classes = [
        TableLoader,
        ParticleIdClassifier,
        CrossValidator,
    ]

    def setup(self):
        """
        Initialize components from config
        """

        self.signal_loader = TableLoader(
            parent=self,
            input_url=self.input_url_signal,
            load_dl1_images=False,
            load_dl1_parameters=True,
            load_dl2=True,
            load_simulated=True,
            load_instrument=True,
        )

        self.background_loader = TableLoader(
            parent=self,
            input_url=self.input_url_background,
            load_dl1_images=False,
            load_dl1_parameters=True,
            load_dl2=True,
            load_simulated=True,
            load_instrument=True,
        )

        if self.signal_loader.subarray != self.background_loader.subarray:
            raise ValueError("Signal and background subarrays do not match")

        self.classifier = ParticleIdClassifier(
            subarray=self.signal_loader.subarray,
            parent=self,
        )
        self.rng = np.random.default_rng(self.random_seed)
        self.cross_validate = CrossValidator(
            parent=self, model_component=self.classifier
        )

    def start(self):
        """
        Train models per telescope type using a cross-validation.
        """
        # By construction both loaders have the same types defined
        types = self.signal_loader.subarray.telescope_types

        self.log.info("Signal input-file: %s", self.signal_loader.input_url)
        self.log.info("Background input-file: %s", self.background_loader.input_url)
        self.log.info("Training models for %d types", len(types))

        for tel_type in types:
            self.log.info("Loading events for %s", tel_type)
            table = self._read_input_data(tel_type)

            self.log.info("Train on %s events", len(table))
            self.cross_validate(tel_type, table)

            self.log.info("Performing final fit for %s", tel_type)
            self.classifier.fit(tel_type, table)
            self.log.info("done")

    def _read_table(self, telescope_type, loader, n_events=None):
        table = loader.read_telescope_events([telescope_type])

        self.log.info("Events read from input: %d", len(table))
        mask = self.classifier.qualityquery.get_table_mask(table)
        table = table[mask]
        self.log.info("Events after applying quality query: %d", len(table))

        table = self.classifier.generate_features(table)

        columns = self.classifier.features + [self.classifier.target]
        table = table[columns]

        valid = check_valid_rows(table)
        if np.any(~valid):
            self.log.warning("Dropping non-predictable events.")
            table = table[valid]

        if n_events is not None:
            n_events = min(n_events, len(table))
            idx = self.rng.choice(len(table), n_events, replace=False)
            idx.sort()
            table = table[idx]

        return table

    def _read_input_data(self, tel_type):
        signal = self._read_table(tel_type, self.signal_loader, self.n_signal)
        background = self._read_table(
            tel_type, self.background_loader, self.n_background
        )
        table = vstack([signal, background])
        self.log.info(
            "Train on %s signal and %s background events", len(signal), len(background)
        )
        return table

    def finish(self):
        """
        Write-out trained models and cross-validation results.
        """
        self.log.info("Writing output")
        self.classifier.write(self.output_path)
        self.signal_loader.close()
        self.background_loader.close()
        if self.cross_validate.output_path:
            self.cross_validate.write()


def main():
    TrainParticleIdClassifier().run()


if __name__ == "__main__":
    main()