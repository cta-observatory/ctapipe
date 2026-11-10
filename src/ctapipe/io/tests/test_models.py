import hashlib
import json
import zipfile
from urllib.parse import unquote

import bagit
import ctao_datamodel as dm
import ctao_datamodel.models.dataproducts as dp
import pytest
from astropy.table import Table
from sklearn.datasets import make_classification
from traitlets.config import Config

from ctapipe.core import Provenance
from ctapipe.core.provenance import json_config_handler
from ctapipe.io.metadata import metadata_to_product, read_ctao_metadata
from ctapipe.io.models import BagItReader, BagItWriter
from ctapipe.reco import ParticleClassifier


# bagit 1.9 does not decode %25 in manifest paths; percent signs are covered below.
@pytest.mark.parametrize("keys", [[], ["LST", "nested/模型\n"]])
def test_bagit_validation(tmp_path, keys):
    path = tmp_path / "models.zip"
    with BagItWriter(path) as writer:
        for key in keys:
            writer.write_joblib(key, {"key": key})

    bag_path = tmp_path / "bag"
    with zipfile.ZipFile(path) as archive:
        archive.extractall(bag_path)

    bag = bagit.Bag(str(bag_path))
    assert bag.validate()
    assert bag.info["model.name"] == "ctapipe/dl2/service/model"
    assert bag.info["data.type"] == "RecoModel"


@pytest.mark.parametrize("keys", [[], ["LST", "nested/模型 %\n"]])
def test_bagit_layout(tmp_path, keys):
    path = tmp_path / "models.zip"
    with BagItWriter(path) as writer:
        for key in keys:
            writer.write_joblib(key, {"key": key})
    writer.close()  # Closing an archive twice must not add duplicate tag files.

    with zipfile.ZipFile(path) as archive:
        assert archive.read("bagit.txt") == (
            b"BagIt-Version: 1.0\nTag-File-Character-Encoding: UTF-8\n"
        )
        assert archive.getinfo("data/").is_dir()
        names = archive.namelist()
        assert len(names) == len(set(names))
        payload_names = {f"data/{key}.pkl" for key in keys}
        assert set(names) == payload_names | {
            "data/",
            "bagit.txt",
            "bag-info.txt",
            "manifest-sha256.txt",
            "tagmanifest-sha256.txt",
        }
        for manifest, expected in [
            ("manifest-sha256.txt", payload_names),
            (
                "tagmanifest-sha256.txt",
                {"bagit.txt", "bag-info.txt", "manifest-sha256.txt"},
            ),
        ]:
            entries = {}
            for line in archive.read(manifest).decode("utf-8").splitlines():
                checksum, name = line.split("  ", 1)
                name = unquote(name)
                assert checksum == hashlib.sha256(archive.read(name)).hexdigest()
                entries[name] = checksum
            assert set(entries) == expected

        metadata = dict(
            line.split(": ", 1)
            for line in archive.read("bag-info.txt").decode("utf-8").splitlines()
        )
        assert "ctao_metadata_version" in metadata
        assert metadata["model.name"] == "ctapipe/dl2/service/model"
        assert metadata["model.version"] == "1.0.0"
        assert not any(key.startswith("CTA ") for key in metadata)

    with BagItReader(path) as reader:
        assert reader.keys == tuple(keys)
        assert reader.meta == metadata_to_product(metadata)
        assert reader.meta.data.level is dp.DataLevel.DL2
        assert reader.meta.data.division is dp.DataDivision.SERVICE
        assert reader.meta.data.association is dp.DataAssociation.SUBARRAY
        assert reader.meta.data.type is dp.DataType.RECO_MODEL
        for key in keys:
            assert reader.read_joblib(key) == {"key": key}


def test_product_metadata(tmp_path):
    config = Config(
        {
            "ProductMetadata": {
                "description": "Reconstruction models: γ showers\ntrained with ctapipe",
                "disclaimer": "For testing only",
                "instance": {"site_id": "CTAO-South", "category": "B"},
                "activity": {"configuration_id": "test:configuration"},
            },
            "Contact": {"name": "Test User", "email": "test@example.org"},
            "Curation": {"release": "test-release"},
        }
    )
    path = tmp_path / "models.zip"
    with (
        Provenance().activity("train-models"),
        BagItWriter(path, config=config) as writer,
    ):
        writer.write_joblib("model", {"value": 1})

    with zipfile.ZipFile(path) as archive:
        info = archive.read("bag-info.txt").decode("utf-8")
        assert (
            "description: Reconstruction models: γ showers\n trained with ctapipe\n"
            in info
        )
        assert "contact.name: Test User\n" in info
        assert "meta.json" not in archive.namelist()

    with BagItReader(path) as reader:
        product = reader.meta
        assert isinstance(product, dp.Product)
        assert (
            product.description
            == "Reconstruction models: γ showers trained with ctapipe"
        )
        assert product.disclaimer == "For testing only"
        assert product.instance.site_id.value == "CTAO-South"
        assert product.instance.category is dp.DataProcessingCategory.B
        assert product.contact.name == "Test User"
        assert product.contact.email == "test@example.org"
        assert product.curation.release == "test-release"
        assert product.activity.configuration_id == "test:configuration"
        assert product.activity.name == "train-models"
        assert product.activity.software.name == "ctapipe"
        assert (
            dm.unflatten_model_instance(
                dm.flatten_model_instance(product), model=dp.Product
            )
            == product
        )

    assert read_ctao_metadata(path) == product


@pytest.mark.parametrize("overwrite", [False, True])
def test_duplicate_model(tmp_path, overwrite):
    with BagItWriter(tmp_path / "models.zip", overwrite=overwrite) as writer:
        writer.write_joblib("LST", 1)
        with pytest.raises(FileExistsError):
            writer.write_joblib("LST", 2)


@pytest.mark.parametrize("key", ["", "../model", "/model", "a/../b", "a\\b", "a\0b"])
def test_invalid_model_key(tmp_path, key):
    with (
        BagItWriter(tmp_path / "models.zip") as writer,
        pytest.raises(ValueError, match="Invalid model key"),
    ):
        writer.write_joblib(key, None)


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
    path = tmp_path / "models.zip"

    with BagItWriter(output_path=path) as writer:
        writer.write_reconstructor(classifier)

        for key, model in classifier._models.items():
            writer.write_joblib(key, model)

    with BagItReader(input_path=path) as reader:
        subarray = reader.subarray
        assert subarray == classifier.subarray
        reconstructor = reader.read_reconstructor()

        # we do not read models by default, we load them one-by-one
        assert len(reconstructor._models) == 0

        assert reader.keys == ("LST", "MST", "SST")
        for key in reader.keys:
            reader.read_joblib(key)

    with zipfile.ZipFile(path) as archive:
        assert "data/subarray.h5" in archive.namelist()
        config = json.loads(archive.read("data/reconstructor_config.json"))
        expected = json.loads(
            json.dumps(classifier.get_current_config(), default=json_config_handler)
        )
        assert config == expected
        entries = archive.read("manifest-sha256.txt").decode().splitlines()
        assert len(entries) == 6  # three models, reconstructor, config and subarray
        for entry in entries:
            checksum, name = entry.split("  ", 1)
            assert checksum == hashlib.sha256(archive.read(name)).hexdigest()

    restored = ParticleClassifier.read(path)
    assert restored.subarray == classifier.subarray
    assert restored._models.keys() == classifier._models.keys()

    # should be the same...
    path = tmp_path / "models2.zip"
    classifier.write(path)

    bag_path = tmp_path / "reconstructor-bag"
    with zipfile.ZipFile(path) as archive:
        archive.extractall(bag_path)
    assert bagit.Bag(str(bag_path)).validate()
