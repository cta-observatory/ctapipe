import numpy as np
from ctapipe.core.tool import Tool, ToolConfigurationError
from ctapipe.core.traits import Path, Int
from ctapipe.io import TableLoader
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm
from astropy.table import vstack

from ..apply import ParticleIdClassifier
from ..preprocessing import check_valid_rows


class TrainParticleIdClassifier(Tool):
    input_url_signal = Path(
        default_value=None,
        allow_none=True,
        directory_ok=False,
    ).tag(config=True)

    input_url_background = Path(
        default_value=None,
        allow_none=True,
        directory_ok=False,
    ).tag(config=True)

    output_path = Path(
        default_value=None,
        allow_none=False,
        directory_ok=False,
    ).tag(config=True)

    n_cross_validation = Int(default_value=5).tag(config=True)
    n_signal = Int(default_value=None, allow_none=True).tag(config=True)
    n_background = Int(default_value=None, allow_none=True).tag(config=True)
    random_seed = Int(default_value=0).tag(config=True)

    aliases = {
        ("s", "signal"): "TrainParticleIdClassifier.input_url_signal",
        ("b", "background"): "TrainParticleIdClassifier.input_url_background",
        ("o", "output"): "TrainParticleIdClassifier.output_path",
    }

    classes = [
        TableLoader,
        ParticleIdClassifier,
    ]

    def setup(self):
        """"""
        if self.input_url_signal is None:
            raise ToolConfigurationError("Need to provide `input_signal_path`")

        if self.input_url_background is None:
            raise ToolConfigurationError("Need to provide `input_signal_path`")

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

    def start(self):
        # By construction both loaders have the same types defined
        types = self.signal_loader.subarray.telescope_types

        self.log.info("Signal input-file: %s", self.signal_loader.input_url)
        self.log.info("Background input-file: %s", self.background_loader.input_url)
        self.log.info("Training models for %d types", len(types))

        for tel_type in types:
            self.log.info("Loading events for %s", tel_type)
            table = self._read_input_data(tel_type)
            self._cross_validate(tel_type, table)

            self.log.info("Performing final fit for %s", tel_type)
            self.classifier.model.fit(tel_type, table)
            self.log.info("done")

    def _read_table(self, telescope_type, loader, n_events=None):
        table = loader.read_telescope_events([telescope_type])

        self.log.info("Events read from input: %d", len(table))
        mask = self.classifier.qualityquery.get_table_mask(table)
        table = table[mask]
        self.log.info("Events after applying quality query: %d", len(table))

        columns = self.classifier.model.features + [self.classifier.target]
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

    def _cross_validate(self, telescope_type, table):
        n_cv = self.n_cross_validation
        self.log.info(f"Starting cross-validation with {n_cv} folds.")

        scores = []

        target = self.classifier.target
        kfold = StratifiedKFold(
            n_splits=n_cv,
            shuffle=True,
            # sklearn does not support numpy's new random API yet
            random_state=self.rng.integers(0, 2**31 - 1),
        )
        cv_it = kfold.split(table, table[target])
        for (train_indices, test_indices) in tqdm(cv_it, total=n_cv):
            train = table[train_indices]
            test = table[test_indices]
            self.classifier.model.fit(telescope_type, train)
            prediction, _ = self.classifier.model.predict_score(telescope_type, test)
            truth = np.where(
                test[target] == self.classifier.model.positive_class,
                1,
                0,
            )
            scores.append(metrics.roc_auc_score(truth, prediction))

        scores = np.array(scores)

        self.log.info(f"Cross validated ROC AUC scores: {scores}")
        self.log.info(
            "Mean ROC AUC score from CV: %s ± %s",
            scores.mean(),
            scores.std(),
        )

    def finish(self):
        self.log.info("Writing output")
        self.classifier.write(self.output_path)
        self.signal_loader.close()
        self.background_loader.close()


def main():
    TrainParticleIdClassifier().run()


if __name__ == "__main__":
    main()
