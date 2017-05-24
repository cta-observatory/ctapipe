from ctapipe.utils import datasets
import os
import pytest

def test_find_datasets():

    # find all datasets matching pattern
    r = datasets.find_all_matching_datasets("(.*)\.camgeom\.fits\.gz")
    assert len(r) > 3

    # get the full filename for a resrouces
    assert os.path.exists(datasets.get_dataset(r[0]))

    # try using a pattern
    r = datasets.find_all_matching_datasets("(.*)\.camgeom\.fits\.gz",
                                            regexp_group=1)
    assert not r[0].endswith("gz")

def test_datasets_in_custom_path(tmpdir_factory):
    """
    check that a dataset in a user-defined CTAPIPE_SVC_PATH is located
    """

    tmpdir1 = tmpdir_factory.mktemp('datasets1')
    tmpdir2 = tmpdir_factory.mktemp('datasets2')
    os.environ['CTAPIPE_SVC_PATH'] = ":".join([str(tmpdir1),str(tmpdir2)])

    # create a dummy dataset to search for:

    dataset_name = "test_dataset_1.txt"
    dataset_path = str(tmpdir1.join(dataset_name))

    with open(dataset_path, "w") as fp:
        fp.write("test test test")

    # try to find dummy dataset
    path = datasets.get_dataset(dataset_name)
    assert path == dataset_path

    with pytest.raises(FileNotFoundError):
        badpath = datasets.get_dataset("does_not_exist")


    # try using find_all_matching_datasets:

    ds = datasets.find_all_matching_datasets("test.*",
                                             searchpath=os.environ['CTAPIPE_SVC_PATH'])
    assert dataset_name in ds