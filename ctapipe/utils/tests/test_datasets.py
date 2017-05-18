from ctapipe.utils import datasets
import os

def test_find_datasets():

    # find all datasets matching pattern
    r = datasets.find_all_matching_datasets("(.*)\.camgeom.fits.gz")
    assert len(r) > 3

    # get the full filename for a resrouces
    assert os.path.exists(datasets.get_dataset(r[0]))

