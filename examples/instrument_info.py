"""
Example of showing some information about the instrumental description
"""

from ctapipe.io.hessio import hessio_event_source
from ctapipe.utils import get_dataset

if __name__ == '__main__':

    # load up one event so that we get the instrument info
    infile = get_dataset("gamma_test.simtel.gz")
    source = hessio_event_source(infile)
    event = next(source)
    del source # close the file

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
