"""
Example of showing some information about the instrumental description
"""

from ctapipe.io import EventSource
from ctapipe.utils import get_dataset_path

if __name__ == "__main__":

    # load up one event so that we get the instrument info
    infile = get_dataset_path("gamma_test_large.simtel.gz")

    source = EventSource(infile)

    print("------ Input: ", infile)

    subarray = source.subarray

    print("\n---------- Subarray Info: -----------")
    subarray.info()
    print("\n---------- Subarray Table:-----------")
    print(subarray.to_table())
    print("\n---------- Subarray Optics:----------")
    print(subarray.to_table(kind="optics"))
    print("\n---------- Mirror Area: -------------")
    print(subarray.tel[1].optics.mirror_area)
