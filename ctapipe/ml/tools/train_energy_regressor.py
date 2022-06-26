import numpy as np
from ctapipe.core.tool import Tool
from ctapipe.core.traits import Path, Unicode, Int
from ctapipe.io import TableLoader
from sklearn import metrics
from sklearn.model_selection import KFold
from tqdm.auto import tqdm

from ..preprocessing import check_valid_rows
from ..sklearn import Regressor


class TrainEnergyRegressor(Tool):

    output_path = Path(
        default_value=None,
        allow_none=False,
        directory_ok=False,
    ).tag(config=True)

    target = Unicode(default_value="true_energy").tag(config=True)
    n_cross_validation = Int(default_value=5).tag(config=True)
    n_events = Int(default_value=None, allow_none=True).tag(config=True)
    random_seed = Int(default_value=0).tag(config=True)

    aliases = {
        ("i", "input"): "TableLoader.input_url",
        ("o", "output"): "TrainEnergyRegressor.output_path",
    }

    classes = [
        TableLoader,
        Regressor,
    ]

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

        self.model = Regressor(
            parent=self,
            target=self.target,
        )
        self.rng = np.random.default_rng(self.random_seed)

    def start(self):

        types = self.loader.subarray.telescope_types
        self.log.info("Inputfile: %s", self.loader.input_url)
        self.log.info("Training models for %d types", len(types))
        for tel_type in types:
            self.log.info("Loading events for %s", tel_type)
            table = self._read_table(tel_type)
            self._cross_validate(tel_type, table)

            self.log.info("Performing final fit for %s", tel_type)
            self.model.fit(tel_type, table)
            self.log.info("done")

    def _read_table(self, telescope_type):
        table = self.loader.read_telescope_events([telescope_type])

        feature_names = self.model.features + [self.target]
        table = table[feature_names]

        valid = check_valid_rows(table)
        self.log.warning("Dropping non-predictable events.")
        table = table[valid]

        if self.n_events is not None:
            n_events = min(self.n_events, len(table))
            idx = self.rng.choice(len(table), n_events, replace=False)
            idx.sort()
            table = table[idx]

        self.log.info("Train on %s events", len(table))

        return table

    def _cross_validate(self, telescope_type, table):
        n_cv = self.n_cross_validation
        self.log.info(f"Starting cross-validation with {n_cv} folds.")

        scores = []

        kfold = KFold(
            n_splits=n_cv,
            shuffle=True,
            # sklearn does not support numpy's new random API yet
            random_state=self.rng.integers(0, 2**31 - 1),
        )

        for (train_indices, test_indices) in tqdm(kfold.split(table), total=n_cv):
            train = table[train_indices]
            test = table[test_indices]

            self.model.fit(telescope_type, train)
            prediction, _ = self.model.predict(telescope_type, test)
            scores.append(metrics.r2_score(test["true_energy"], prediction))

        scores = np.array(scores)

        self.log.info(f"Cross validated R^2 scores: {scores}")
        self.log.info(
            "Mean R^2 score from CV: %s ± %s",
            scores.mean(),
            scores.std(),
        )

    def finish(self):
        self.log.info("Writing output")
        self.model.write(self.output_path)
        self.loader.close()


def main():
    TrainEnergyRegressor().run()


if __name__ == "__main__":
    main()
