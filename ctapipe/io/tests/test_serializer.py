from copy import deepcopy
from os import remove

import pytest
from astropy.io import fits

from ctapipe.io.hessio import hessio_event_source
from ctapipe.io.serializer import Serializer
from ctapipe.io.sources import PickleSource
from ctapipe.utils import get_dataset_path


def compare(read_container, source_container):
    # test if 4th adc value of telescope 17 HI_GAIN are equals
    return (read_container.r0.tel[17].waveform[0][2][4] ==
            source_container.r0.tel[17].waveform[0][2][4])


def generate_input_containers():
    # Get event from hessio file, append them into input_containers
    input_filename = get_dataset_path("gamma_test.simtel.gz")
    gen = hessio_event_source(input_filename, max_events=3)
    input_containers = []
    for event in gen:
        input_containers.append(deepcopy(event))
    return input_containers

# Setup
input_containers = generate_input_containers()


@pytest.fixture(scope='session')
def binary_filename(tmpdir_factory):
    return str(tmpdir_factory.mktemp('data')
               .join('pickle_data.pickle.gz'))


@pytest.fixture(scope='session')
def fits_file_name(tmpdir_factory):
    return str(tmpdir_factory.mktemp('data').join('output.fits'))



def test_pickle_serializer(binary_filename):
    serial = Serializer(filename=binary_filename, format='pickle', mode='w')
    # append all input file events in input_containers list and pickle serializer
    for event in input_containers:
        serial.add_container(event)
    serial.close()

    # read Containers from pickle serializer
    reader = PickleSource(filename=binary_filename)
    # file read_containers from serializer generator
    read_containers = []
    for container in reader:
        read_containers.append(container)
    # test if number of read Container correspond to input
    assert len(read_containers) is len(input_containers)
    # test if 4th adc value of telescope 17 HI_GAIN are equals
    assert compare(input_containers[2], read_containers[2])
    reader.close()
    remove(binary_filename)


# Test pickle reader/writer with statement
def test_pickle_with_statement(binary_filename):
    with Serializer(filename=binary_filename, format='pickle', mode='w') as \
            containers_writer:
        for container in input_containers:
            containers_writer.add_container(container)
        containers_writer.close()

    read_containers = []
    with PickleSource(filename=binary_filename) as reader:
        for container in reader:
            read_containers.append(container)
    # test if number of read Container correspond to input
    assert len(read_containers) is len(input_containers)
    # test if 4th adc value of telescope 17 HI_GAIN are equals
    assert compare(input_containers[2], read_containers[2])
    remove(binary_filename)


# Test pickle reader iterator
def test_pickle_iterator(binary_filename):
    serial = Serializer(filename=binary_filename, format='pickle',
                        mode='w')
    # append all events in input_containers list and pickle serializer
    for event in input_containers:
        serial.add_container(event)
    serial.close()

    read_containers = []
    reader = PickleSource(filename=binary_filename)
    for container in reader:
        read_containers.append(container)
    # test if number of read Container correspond to input
    assert len(read_containers) is len(input_containers)
    # test if 4th adc value of telescope 17 HI_GAIN are equals
    assert compare(input_containers[2], read_containers[2])
    reader.close()
    remove(binary_filename)





def test_fits_dl0(fits_file_name):
    serial = Serializer(filename=fits_file_name, format='fits', mode='w')
    for container in input_containers:
        serial.add_container(container.dl0)
    serial.close()
    hdu = fits.open(fits_file_name)[1]
    assert hdu.data["event_id"][0] == 408
    assert hdu.data["event_id"][1] == 409
    assert hdu.data["event_id"][2] == 803
    assert hdu.data["obs_id"][2] == 31964
    remove(fits_file_name)


def test_exclusive_mode(fits_file_name):
    serial = Serializer(filename=fits_file_name, format='fits', mode='w')
    for container in input_containers:
        serial.add_container(container.dl0)
    serial.close()
    # Try to write to fits_file_name in exclusive mode
    with pytest.raises(OSError):
        serial = Serializer(filename=fits_file_name, format='fits', mode='x')
        serial.add_container(input_containers[2].dl0)
        serial.close()
    remove(fits_file_name)

"""
def test_fits_dl1():
    input_test_file = get_datasets_path('example_container.pickle.gz')
    with gzip_open(input_test_file, 'rb') as f:
        data = load(f)
    t38 = data[0].dl1.tel[38]
    serial = Serializer('output.fits', 'fits', overwrite=True)
    serial.add_container(t38)
    serial.write()
    # t11_1 = data[1].dl1.tel[11]
    # S_cal.write(t11_1) # This will not work because shape of data is different from tel to tel.
"""


def test_fits_context_manager(fits_file_name):
    with Serializer(filename=fits_file_name, format='fits', mode='w') as writer:
        for container in input_containers:
            writer.add_container(container.dl0)

    hdulist = fits.open(fits_file_name)
    assert hdulist[1].data["event_id"][0] == 408
    remove(fits_file_name)


# TODO test FITSSource class
