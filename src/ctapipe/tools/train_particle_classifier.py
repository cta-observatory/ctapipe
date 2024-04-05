"""
Tool for training the ParticleClassifier
"""
import numpy as np
from astropy.table import vstack

from ctapipe.core.tool import Tool, ToolConfigurationError
from ctapipe.core.traits import Float, Int, IntTelescopeParameter, Path
from ctapipe.io import TableLoader
from ctapipe.reco import CrossValidator, ParticleClassifier

from .utils import read_training_events

__all__ = [
    "TrainParticleClassifier",
]


class TrainParticleClassifier(Tool):
    """
    Tool to train a `~ctapipe.reco.ParticleClassifier` on dl1b/dl2 data.

    The tool first performs a cross validation to give an initial estimate
    on the quality of the estimation and then finally trains one model
    per telescope type on the full dataset.
    """

    name = "ctapipe-train-particle-classifier"
    description = __doc__

    examples = """
    ctapipe-train-particle-classifier \\
        -c train_particle_classifier.yaml \\
        --signal gamma.dl2.h5 \\
        --background proton.dl2.h5 \\
        -o particle_classifier.pkl
    """

    input_url_signal = Path(
        default_value=None,
        allow_none=False,
        directory_ok=False,
        help="Input dl1b/dl2 file for the signal class.",
    ).tag(config=True)

    input_url_background = Path(
        default_value=None,
        allow_none=False,
        directory_ok=False,
        help="Input dl1b/dl2 file for the background class.",
    ).tag(config=True)

    output_path = Path(
        default_value=None,
        allow_none=False,
        directory_ok=False,
        help=(
            "Output file for the trained reconstructor."
            " At the moment, pickle is the only supported format."
        ),
    ).tag(config=True)

    n_events = IntTelescopeParameter(
        default_value=None,
        allow_none=True,
        help=(
            "Total number of events to be used for training."
            " If not given, all available events will be used"
            " (considering ``signal_fraction``)."
        ),
    ).tag(config=True)

    signal_fraction = Float(
        default_value=0.5,
        allow_none=False,
        help=(
            "Fraction of signal events in all events to be used for training."
            " ``signal_fraction`` = n_signal / (n_signal + n_background)"
        ),
    ).tag(config=True)

    chunk_size = Int(
        default_value=100000,
        allow_none=True,
        help=(
            "How many subarray events to load at once before training on"
            " n_events (or all available) events."
        ),
    ).tag(config=True)

    random_seed = Int(
        default_value=0, help="Random seed for sampling training events."
    ).tag(config=True)

    n_jobs = Int(
        default_value=None,
        allow_none=True,
        help="Number of threads to use for the reconstruction. This overwrites the values in the config of each reconstructor.",
    ).tag(config=True)

    aliases = {
        "signal": "TrainParticleClassifier.input_url_signal",
        "background": "TrainParticleClassifier.input_url_background",
        "n-events": "TrainParticleClassifier.n_events",
        "signal-fraction": "TrainParticleClassifier.signal_fraction",
        "n-jobs": "ParticleClassifier.n_jobs",
        ("o", "output"): "TrainParticleClassifier.output_path",
        "cv-output": "CrossValidator.output_path",
    }

    classes = [
        TableLoader,
        ParticleClassifier,
        CrossValidator,
    ]

    def setup(self):
        """
        Initialize components from config.
        """
        self.signal_loader = self.enter_context(
            TableLoader(
                parent=self,
                input_url=self.input_url_signal,
            )
        )
        self.background_loader = self.enter_context(
            TableLoader(
                parent=self,
                input_url=self.input_url_background,
            )
        )
        if self.signal_loader.subarray != self.background_loader.subarray:
            raise ValueError("Signal and background subarrays do not match")

        self.n_events.attach_subarray(self.signal_loader.subarray)
        self.classifier = ParticleClassifier(
            subarray=self.signal_loader.subarray, parent=self
        )
        self.cross_validate = self.enter_context(
            CrossValidator(
                parent=self, model_component=self.classifier, overwrite=self.overwrite
            )
        )
        self.rng = np.random.default_rng(self.random_seed)
        self.check_output(self.output_path)

    def start(self):
        """
        Train models per telescope type.
        """
        # By construction both loaders have the same types defined
        types = self.signal_loader.subarray.telescope_types

        self.log.info("Signal input-file: %s", self.signal_loader.input_url)
        self.log.info("Background input-file: %s", self.background_loader.input_url)
        self.log.info("Training models for %d types", len(types))

        for tel_type in types:
            self.log.info("Loading events for %s", tel_type)
            table = self._read_input_data(tel_type)
            self.cross_validate(tel_type, table)

            self.log.info("Performing final fit for %s", tel_type)
            self.classifier.fit(tel_type, table)
            self.log.info("done")

    def _read_input_data(self, tel_type):
        if self.signal_fraction < 0 or self.signal_fraction > 1:
            raise ToolConfigurationError(
                "The signal_fraction has to be between 0 and 1"
            )

        feature_names = self.classifier.features + [
            self.classifier.target,
            "true_energy",
            "true_impact_distance",
        ]
        n_events = self.n_events.tel[tel_type]
        if n_events is not None:
            n_signal = int(self.signal_fraction * n_events)
            n_background = n_events - n_signal
        else:
            n_signal = None
            n_background = None

        signal = read_training_events(
            loader=self.signal_loader,
            chunk_size=self.chunk_size,
            telescope_type=tel_type,
            reconstructor=self.classifier,
            feature_names=feature_names,
            rng=self.rng,
            log=self.log,
            n_events=n_signal,
        )
        background = read_training_events(
            loader=self.background_loader,
            chunk_size=self.chunk_size,
            telescope_type=tel_type,
            reconstructor=self.classifier,
            feature_names=feature_names,
            rng=self.rng,
            log=self.log,
            n_events=n_background,
        )
        if n_events is None:  # use as many events as possible (keeping signal_fraction)
            n_signal = len(signal)
            n_background = len(background)

            if n_signal < (n_signal + n_background) * self.signal_fraction:
                n_background = int(n_signal * (1 / self.signal_fraction - 1))
                self.log.info("Sampling %d background events", n_background)
                idx = self.rng.choice(len(background), n_background, replace=False)
                idx.sort()
                background = background[idx]
            else:
                n_signal = int(n_background / (1 / self.signal_fraction - 1))
                self.log.info("Sampling %d signal events", n_signal)
                idx = self.rng.choice(len(signal), n_signal, replace=False)
                idx.sort()
                signal = signal[idx]

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
        self.classifier.n_jobs = None
        self.classifier.write(self.output_path, overwrite=self.overwrite)
        self.signal_loader.close()
        self.background_loader.close()
        self.cross_validate.close()


def main():
    TrainParticleClassifier().run()


if __name__ == "__main__":
    main()
