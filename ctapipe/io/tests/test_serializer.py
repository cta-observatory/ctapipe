from ctapipe.io.serializer import PickleSerializer
from ctapipe.io.hessio import hessio_event_source
from ctapipe.utils.datasets import get_datasets_path
from os import remove


def test_pickle_serializer():
    binary_filename = 'pickle_data.pickle.gzip'
    input_filename = get_datasets_path("gamma_test.simtel.gz")
    gen = hessio_event_source(input_filename, max_events=3)
    input_containers = []
    serial = PickleSerializer(binary_filename, overwrite=True)
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
    with PickleSerializer('pickle_data.pickle.gzip') as containers:
        for _ in containers:
            pass
    remove('pickle_data.pickle.gzip')


def test_pickle_serializer_file_extension():
    serial = PickleSerializer("pickle_serializer", overwrite=True)
    output_file = serial.write()
    assert output_file.find('.pickle.gzip')
    remove(output_file)

