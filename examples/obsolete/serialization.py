#!/usr/bin/env python3

from ctapipe.io.serializer import Serializer
from astropy import log
import pickle
import gzip

log.setLevel('DEBUG')


def write_dl0_example(filename, data):
    S = Serializer(filename, mode='w')

    # Create table
    for container in data:
        S.write(container.dl0)

    # Print table
    print(S._writer.table)

    # Save table to disk
    S.save()
    return S


def write_dl1_tel_example(filename, data):

    t38 = data[0].dl1.tel[38]

    S_cal = Serializer(filename, mode='w')
    S_cal.write(t38)

    print(S_cal._writer.table)

    # t11_1 = data[1].dl1.tel[11]
    # S_cal.write(t11_1)
    # This will not work because shape of data is different from tel to tel.

    S_cal.save()
    return S_cal


def context_manager_example(filename, data):
    with Serializer(filename, mode='w') as writer:
        for container in data:
            print(container.dl0)
            writer.write(container.dl0)
            print(writer._writer.table)
    return 0


if __name__ == "__main__":

    with gzip.open('example_container.pickle.gz', 'rb') as f:
        data = pickle.load(f)

    S = write_dl0_example("output.fits", data)
    S_cal = write_dl1_tel_example("cal_output.fits", data)
    S_context = context_manager_example("output_context.fits", data)
