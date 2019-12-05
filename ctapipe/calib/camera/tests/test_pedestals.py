import numpy as np
from ctapipe.calib.camera.pedestals import *
from ctapipe.io.containers import EventAndMonDataContainer
import astropy.units as u
from ctapipe.instrument import SubarrayDescription, TelescopeDescription


def test_pedestal_calculator():
    """ test of PedestalIntegrator """

    tel_id = 0
    n_events = 10
    n_gain = 2
    n_pixels = 1855
    ped_level = 300

    subarray = SubarrayDescription(
        "test array",
        tel_positions={0: np.zeros(3) * u.m},
        tel_descriptions={
            0: TelescopeDescription.from_name(
                optics_name="SST-ASTRI", camera_name="CHEC"
            ),
        }
    )

    ped_calculator = PedestalIntegrator(
        subarray=subarray,
        charge_product="FixedWindowSum",
        sample_size=n_events,
        tel_id=tel_id
    )
    # create one event
    data = EventAndMonDataContainer()
    data.meta['origin'] = 'test'

    # fill the values necessary for the pedestal calculation
    data.mon.tel[tel_id].pixel_status.hardware_failing_pixels = np.zeros((n_gain, n_pixels), dtype=bool)
    data.r1.tel[tel_id].waveform = np.full((2, n_pixels, 40), ped_level)
    data.r1.tel[tel_id].trigger_time = 1000

    while ped_calculator.num_events_seen < n_events :
        if ped_calculator.calculate_pedestals(data):
            assert data.mon.tel[tel_id].pedestal
            assert np.mean(data.mon.tel[tel_id].pedestal.charge_median) == (
                    ped_calculator.extractor.window_width[0] * ped_level)
            assert np.mean(data.mon.tel[tel_id].pedestal.charge_std) == 0


def test_calc_pedestals_from_traces():
    """ test calc_pedestals_from_traces """
    # create some test data (all ones, but with a 2 stuck in for good measure):
    npix = 1000
    nsamp = 32
    traces = np.ones(shape=(npix, nsamp), dtype=np.int32)
    traces[:, 2] = 2

    # calculate pedestals over samples 0-10
    peds, pedvars = calc_pedestals_from_traces(traces, 0, 10)

    # check the results. All pedestals should be 1.1 and vars 0.09:

    assert np.all(peds == 1.1)
    assert np.all(pedvars == 0.09)
    assert peds.shape[0] == npix
    assert pedvars.shape[0] == npix

    # try another sample range, where there should be no variances and
    # all 1.0 peds:
    peds, pedvars = calc_pedestals_from_traces(traces, 12, 32)

    assert np.all(peds == 1.0)
    assert np.all(pedvars == 0)
