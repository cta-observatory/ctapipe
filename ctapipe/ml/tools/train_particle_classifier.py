import numpy as np
from ctapipe.core.tool import Tool
from ctapipe.core.traits import Path, Unicode, Int
from ctapipe.io import TableLoader
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm
from astropy.table import vstack

from ..preprocessing import check_valid_rows
from ..sklearn import Classifier


class TrainParticleIdClassifier(Tool):
    input_signal_path = Path(
        default_value=None,
        allow_none=False,
        directory_ok=False,
    ).tag(config=True)
    input_background_path = Path(
        default_value=None,
        allow_none=False,
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
        ("input-background"): "TrainParticleIdClassifier.input_background_path",
        ("input-signal"): "TrainParticleIdClassifier.input_signal_path",
        ("o", "output"): "TrainParticleIdClassifier.output_path",
    }

    classes = [
        TableLoader,
        Classifier,
    ]

    def setup(self):
        """"""
        self.signal_loader = TableLoader(
            parent=self,
            input_url=self.input_signal_path,
            load_dl1_images=False,
            load_dl1_parameters=True,
            load_dl2=True,
            load_simulated=True,
            load_instrument=True,
        )
        self.background_loader = TableLoader(
            parent=self,
            input_url=self.input_background_path,
            load_dl1_images=False,
            load_dl1_parameters=True,
            load_dl2=True,
            load_simulated=True,
            load_instrument=True,
        )
        self.model = Classifier(
            parent=self,
            target="hadronness",
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
            self.model.fit(tel_type, table)
            self.log.info("done")

    def _read_table(self, telescope_type, loader, n_events=None):
        table = loader.read_telescope_events([telescope_type])
        table["hadronness"] = (table["true_shower_primary_id"] != 0).astype(np.int8)
        table = table[self.model.features + ["hadronness"]]

        valid = check_valid_rows(table)
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

        kfold = StratifiedKFold(
            n_splits=n_cv,
            shuffle=True,
            # sklearn does not support numpy's new random API yet
            random_state=self.rng.integers(0, 2**31 - 1),
        )

        for (train_indices, test_indices) in tqdm(
            kfold.split(table, table["hadronness"]), total=n_cv
        ):
            train = table[train_indices]
            test = table[test_indices]
            self.model.fit(telescope_type, train)
            prediction, _ = self.model.predict(telescope_type, test)
            scores.append(metrics.roc_auc_score(test["hadronness"], prediction))

        scores = np.array(scores)

        self.log.info(f"Cross validated ROC AUC scores: {scores}")
        self.log.info(
            "Mean ROC AUC score from CV: %s ± %s",
            scores.mean(),
            scores.std(),
        )

    def finish(self):
        self.log.info("Writing output")
        self.model.write(self.output_path)
        self.signal_loader.close()
        self.background_loader.close()


def main():
    TrainParticleIdClassifier().run()


if __name__ == "__main__":
    main()
