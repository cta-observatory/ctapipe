from ctapipe.io.data_container import create_dummy_chunk

def test_iteration():
    #create 1 chunk and iterate over events
    for event in create_dummy_chunk():
        print(event)

def test_length():
    #test whether the CameraData class overwrites __len__ correctly
    chunk = create_dummy_chunk(N=128)
    assert len(chunk) == 128

def test_camera_description_contents():
    chunk = create_dummy_chunk(N=10)
    desc = chunk.camera_description
    assert desc != None

    print(desc)


def test_dimensions():
    N =10
    pixels = 40*40
    samples = 50
    chunk = create_dummy_chunk(N=N, pixels=pixels, samples=samples)
    chunk.table.pprint(max_width=-1)

    assert chunk.adc_sums.shape == (N, pixels), 'adc_sums does not have the correct dimensions.'

    assert chunk.adc_samples.shape == (N, pixels, samples), 'adc_samples does not have the correct dimensions.'
