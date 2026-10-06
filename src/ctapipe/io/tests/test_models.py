import pytest
from sklearn.datasets import make_classification

from ctapipe.reco import ParticleClassifier


@pytest.fixture(scope="session")
def models():
    from sklearn.ensemble import RandomForestClassifier

    models = {}
    for key in ("LST", "MST", "SST"):
        X, y = make_classification(n_samples=100)

        clf = RandomForestClassifier()
        clf.fit(X, y)

        models[key] = clf

    return models


def test_write_reconstructor(tmp_path, subarray_prod5_paranal, models):
    from ctapipe.io.models import SklearnModelWriter

    path = tmp_path / "models.zip"

    clf = ParticleClassifier(
        subarray=subarray_prod5_paranal, model_cls="RandomForestClassifier"
    )

    with SklearnModelWriter(output_path=path) as writer:
        writer.write_reconstructor_config(clf)
        writer.write_subarray(subarray_prod5_paranal)

        for key, model in models.items():
            writer(key, model)
