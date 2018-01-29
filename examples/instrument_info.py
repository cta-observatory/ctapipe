"""
Example of showing some information about the instrumental description
"""

from ctapipe.io import event_source
from ctapipe.utils import get_dataset

if __name__ == '__main__':

    # load up one event so that we get the instrument info
    infile = get_dataset("gamma_test.simtel.gz")
    with event_source(infile) as source:
        gsource = (x for x in source)
        event = next(gsource)

    print("------ Input: ", infile)
    
    subarray = event.inst.subarray

    print("\n---------- Subarray Info: -----------")
    subarray.info()
    print("\n---------- Subarray Table:-----------")
    print(subarray.to_table())
    print("\n---------- Subarray Optics:----------")
    print(subarray.to_table(kind='optics'))
    print("\n---------- Mirror Area: -------------")
    print(subarray.tel[1].optics.mirror_area)
