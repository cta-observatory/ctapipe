import pytest
from astropy.table import Table
from sklearn.datasets import make_classification

from ctapipe.reco import ParticleClassifier


@pytest.fixture(scope="session")
def classifier(subarray_prod5_paranal):

    features = [f"col{i}" for i in range(10)]
    clf = ParticleClassifier(
        subarray=subarray_prod5_paranal,
        model_cls="RandomForestClassifier",
        features=features,
    )

    for key in ("LST", "MST", "SST"):
        X, y = make_classification(
            n_samples=10000, n_features=len(features), n_classes=2
        )

        table = Table({col: X[:, i] for i, col in enumerate(features)})
        table["true_shower_primary_id"] = y
        clf.fit(key, table)

    return clf


def test_reconstructor_io(tmp_path, classifier):
    from ctapipe.io.models import ZipModelReader, ZipModelWriter

    path = tmp_path / "models.zip"

    with ZipModelWriter(output_path=path) as writer:
        writer.write_reconstructor(classifier)

        for key, model in classifier._models.items():
            writer(key, model)

    with ZipModelReader(input_path=path) as reader:
        subarray = reader.subarray
        assert subarray == classifier.subarray

        reconstructor = reader.read_compressed_joblib("reconstructor")

        # we do not read models by default, we load them one-by-one
        assert len(reconstructor._models) == 0

    # should be the same...
    path = tmp_path / "models2.zip"
    classifier.write(path)
