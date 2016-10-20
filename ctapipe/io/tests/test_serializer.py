from ctapipe.io.serializer import PickleSerializer
from ctapipe.io.serializer import FitsSerializer
from ctapipe.io.hessio import hessio_event_source
from ctapipe.utils.datasets import get_datasets_path
from os import remove
import gzip
from pickle import load

# PICKLE SERIALIZER


def test_pickle_serializer():
    binary_filename = 'pickle_data.pickle.gzip'
    input_filename = get_datasets_path("gamma_test.simtel.gz")
    gen = hessio_event_source(input_filename, max_events=3)
    input_containers = []
    serial = PickleSerializer(file=binary_filename, overwrite=True)
#   add all input file events in  input_containers list and pickle serializer
    for event in gen:
        serial.add_container(event)
        input_containers.append(event)
    # write Containers to pickle serializer
    output_file = serial.write()

    # read Containers from pickle serializer
    serial = PickleSerializer(output_file)
    # file read_containers from serializer generator
    read_containers = []
    for event in serial:
        read_containers.append(event)
#   test if number of read Container correspond to input
    assert len(read_containers) is len(input_containers)

    event_in = input_containers[1]
    event_read = read_containers[1]

#    test if 4th adc value of telescope 17 HI_GAIN are equals
    assert (event_in.dl0.tel[17].adc_samples[0][2][4] ==
            event_read.dl0.tel[17].adc_samples[0][2][4])


def test_pickle_serializer_with_statement():
    with PickleSerializer(file='pickle_data.pickle.gzip') as containers:
        for _ in containers:
            pass
    remove('pickle_data.pickle.gzip')


def test_pickle_serializer_file_extension():
    serial = PickleSerializer(file="pickle_serializer", overwrite=True)
    output_file = serial.write()
    assert output_file.find('.pickle.gzip')
    remove(output_file)


def test_pickle_serializer_traitlets():
    serial = PickleSerializer()
    assert serial.has_trait('file') is True
    serial.file = 'pickle_data.pickle.gzip'


# FITS SERIALIZER

def test_fits_dl0():
    """
    input_test_file = get_datasets_path('example_container.pickle.gz')
    with gzip.open(input_test_file, 'rb') as f:
        data = load(f)
    """
    input_filename = get_datasets_path("gamma_test.simtel.gz")
    gen = hessio_event_source(input_filename, max_events=3)
    data = next(gen)
    serial = FitsSerializer('output.fits', 'fits', overwrite=True)
    container = data.dl0
    serial.add_container(container)
    serial.write()

"""
def test_fits_dl1():
    input_test_file = get_datasets_path('example_container.pickle.gz')
    with gzip.open(input_test_file, 'rb') as f:
        data = load(f)
    t38 = data[0].dl1.tel[38]
    serial = FitsSerializer('output.fits', 'fits', overwrite=True)
    serial.add_container(t38)
    serial.write()
    # t11_1 = data[1].dl1.tel[11]
    # S_cal.write(t11_1) # This will not work because shape of data is different from tel to tel.
"""

def test_fits_context_manager():
    input_filename = get_datasets_path("gamma_test.simtel.gz")
    data = hessio_event_source(input_filename, max_events=3)

    with FitsSerializer('output.fits', 'fits', overwrite=True) as writer:
        for container in data:
            writer.add_container(container.dl0)
        writer.write()


