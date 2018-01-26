from ctapipe.utils import get_dataset
from ctapipe.io.hessioeventsource import HESSIOEventSource


def test_hessio_file_reader():
    dataset = get_dataset("gamma_test.simtel.gz")
    kwargs = dict(config=None, tool=None, input_url=dataset)
    with HESSIOEventSource(**kwargs) as reader:
        assert reader.is_compatible(dataset)
        assert not reader.is_stream
        for event in reader:
            if event.count == 0:
                assert event.r0.tels_with_data == {38, 47}
            elif event.count == 1:
                assert event.r0.tels_with_data == {11, 21, 24, 26, 61, 63, 118,
                                                   119}
            else:
                break
        for event in reader:
            # Check generator has restarted from beginning
            assert event.count == 0
            break

    # test that max_events works:
    max_events = 5
    with HESSIOEventSource(**kwargs, max_events=max_events) as reader:
        count = 0
        for _ in reader:
            count += 1
        assert count == max_events

    # test that the allowed_tels mask works:
    with HESSIOEventSource(**kwargs, allowed_tels={3, 4}) as  reader:
        for event in reader:
            assert event.r0.tels_with_data.issubset(reader.allowed_tels)

