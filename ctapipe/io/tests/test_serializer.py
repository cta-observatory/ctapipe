from ctapipe.io.serializer import Serializer
from ctapipe.io.hessio import hessio_event_source
from ctapipe.utils.datasets import get_datasets_path
from os import remove


# PICKLE SERIALIZER
binary_filename = 'pickle_data.pickle.gz'
input_containers = []


def compare(read_container, source_container):
    return (read_container.dl0.tel[17].adc_samples[0][2][4] ==
            source_container.dl0.tel[17].adc_samples[0][2][4])


def test_prepare_input_data():
    # Get event from hessio file, append them into input_containers which will
    # be used later for comparison
    input_filename = get_datasets_path("gamma_test.simtel.gz")
    gen = hessio_event_source(input_filename, max_events=3)
    #  append all input file events in  input_containers
    #  list and pickle serializer
    for event in gen:
        input_containers.append(event)


def test_pickle_serializer():
    serial = Serializer(filename=binary_filename, format='pickle',
                        mode='w')
#   append all input file events in  input_containers list and pickle serializer
    for event in input_containers:
        serial.add_container(event)
    serial.write()

    # read Containers from pickle serializer
    serial = Serializer(filename=binary_filename, format='pickle', mode='r')
    # file read_containers from serializer generator
    read_containers = []
    while True:
        try:
            read_containers.append(serial.get_next_container())
        except EOFError:
            break
#   test if number of read Container correspond to input
    assert len(read_containers) is len(input_containers)
#   test if 4th adc value of telescope 17 HI_GAIN are equals
    assert compare(input_containers[2], read_containers[2])
    serial.close()

# Test pickle reader/writer with statement
def test_pickle_with_statement():

    with Serializer(filename=binary_filename, format='pickle', mode='w') as \
            containers_writer:
        for container in input_containers:
            containers_writer.add_container(container)
        containers_writer.write()

    read_containers = []
    with Serializer(filename=binary_filename, format='pickle', mode='r') as\
            containers_reader:
        while True:
            try:
                read_containers.append(containers_reader.get_next_container())
            except EOFError:
                break
    #   test if number of read Container correspond to input
    assert len(read_containers) is len(input_containers)
    #   test if 4th adc value of telescope 17 HI_GAIN are equals
    assert compare(input_containers[2], read_containers[2])
    containers_reader.close()


# Test pickle reader iterator
def test_pickle_iterator():
    read_containers = []
    serial = Serializer(filename=binary_filename, format='pickle', mode='r')
    for container in serial:
        read_containers.append(container)
    #   test if number of read Container correspond to input
    assert len(read_containers) is len(input_containers)
    #   test if 4th adc value of telescope 17 HI_GAIN are equals
    assert compare(input_containers[2], read_containers[2])


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
    serial = Serializer(filename='output.fits', format='fits', mode='w')
    container = data.dl0
    serial.add_container(container)
    serial.write()

"""
def test_fits_dl1():
    input_test_file = get_datasets_path('example_container.pickle.gz')
    with gzip_open(input_test_file, 'rb') as f:
        data = load(f)
    t38 = data[0].dl1.tel[38]
    serial = FitsSerializer('output.fits', 'fits', overwrite=True)
    serial.add_container(t38)
    serial.write()
    # t11_1 = data[1].dl1.tel[11]
    # S_cal.write(t11_1) # This will not work because shape of data is different from tel to tel.
"""


def fits_context_manager():
    with Serializer(filename='output.fits', format='fits', mode='w') as writer:
        for container in input_containers:
            writer.add_container(container.dl0)
        writer.write()

# Remove produced files during tests
def test_remove_test_file():
    remove(binary_filename)
