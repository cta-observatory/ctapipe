import numpy as np
from ctapipe.core.tool import Tool
from ctapipe.core.traits import Path, Unicode
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

    aliases = {
        ("i", "input"): "TableLoader.input_url",
        ("o", "output"): "TrainEnergyRegressor.output_path",
        "model": "TrainEnergyRegressor.Regressor.model_cls",
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
            load_dl2_geometry=True,
            load_simulated=True,
        )

    def start(self):
        self.model = Regressor(
            parent=self,
            target=self.target,
        )

        table = self._read_table()

        self._cross_validate(table)

        self.model.fit(table)

    def _read_table(self):
        table = self.loader.read_telescope_events()

        feature_names = self.model.features + [self.target]
        table = table[feature_names]

        valid = check_valid_rows(table)
        self.log.warning("Dropping not-predictable events.")
        table = table[valid]

        self.log.info("Train on %s events", len(table))

        return table

    def _cross_validate(self, table):
        n_cv = 5
        self.log.info(f"Starting cross-validation with {n_cv} folds.")

        scores = []

        kfold = KFold(n_splits=n_cv, shuffle=True, random_state=42)

        for (train_indices, test_indices) in tqdm(kfold.split(table), total=n_cv):
            train = table[train_indices]
            test = table[test_indices]

            self.model.fit(train)

            prediction = self.model.predict(test)

            scores.append(metrics.r2_score(test["true_energy"], prediction))

        scores = np.array(scores)

        self.log.info(f"Cross validated R^2 scores: {scores}")
        self.log.info(
            "Mean R^2 score from CV: %s ± %s",
            scores.mean(),
            scores.std(),
        )

    def finish(self):
        self.model.write(self.output_path)
        self.loader.close()


if __name__ == "__main__":
    TrainEnergyRegressor().run()
