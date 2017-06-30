"""
Example of showing some information about the instrumental description
"""

from ctapipe.io.hessio import hessio_event_source


if __name__ == '__main__':

    # load up one event so that we get the instrument info
    source = hessio_event_source("/Users/kosack/Data/CTA/Prod3/gamma.simtel.gz")
    event = next(source)
    del source # close the file


    subarray = event.inst.subarray
    subarray.info()
    print(subarray.to_table())


