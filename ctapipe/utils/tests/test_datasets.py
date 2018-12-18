import json
import os

import pytest
import yaml

from ctapipe.utils import datasets


def test_find_datasets():
    # find all datasets matching pattern
    r = datasets.find_all_matching_datasets(r"(.*)\.camgeom\.fits\.gz")
    assert len(r) > 3

    # get the full filename for a resrouces
    assert os.path.exists(datasets.get_dataset_path(r[0]))

    # try using a pattern
    r = datasets.find_all_matching_datasets(r"(.*)\.camgeom\.fits\.gz",
                                            regexp_group=1)
    assert not r[0].endswith("gz")


def test_datasets_in_custom_path(tmpdir_factory):
    """
    check that a dataset in a user-defined CTAPIPE_SVC_PATH is located
    """

    tmpdir1 = tmpdir_factory.mktemp('datasets1')
    tmpdir2 = tmpdir_factory.mktemp('datasets2')
    os.environ['CTAPIPE_SVC_PATH'] = ":".join([str(tmpdir1), str(tmpdir2)])

    # create a dummy dataset to search for:

    dataset_name = "test_dataset_1.txt"
    dataset_path = str(tmpdir1.join(dataset_name))

    with open(dataset_path, "w") as fp:
        fp.write("test test test")

    # try to find dummy dataset
    path = datasets.get_dataset_path(dataset_name)
    assert path == dataset_path

    with pytest.raises(FileNotFoundError):
        datasets.get_dataset_path("does_not_exist")

    # try using find_all_matching_datasets:

    ds = datasets.find_all_matching_datasets("test.*",
                                             searchpath=os.environ[
                                                 'CTAPIPE_SVC_PATH'])
    assert dataset_name in ds


def test_structured_datasets(tmpdir):
    basename = "test.yml"

    test_data = dict(x=[1, 2, 3, 4, 5], y='test_json')

    os.environ['CTAPIPE_SVC_PATH'] = ":".join([str(tmpdir)])

    with tmpdir.join("data_test.json").open(mode='w') as fp:
        json.dump(test_data, fp)

    data1 = datasets.get_structured_dataset('data_test')
    assert data1['x'] == [1, 2, 3, 4, 5]
    assert data1['y'] == 'test_json'
    tmpdir.join("data_test.json").remove()

    test_data['y'] = 'test_yaml'
    with tmpdir.join("data_test.yaml").open(mode='w') as fp:
        yaml.dump(test_data, fp)

    data1 = datasets.get_structured_dataset('data_test')
    assert data1['x'] == [1, 2, 3, 4, 5]
    assert data1['y'] == 'test_yaml'
