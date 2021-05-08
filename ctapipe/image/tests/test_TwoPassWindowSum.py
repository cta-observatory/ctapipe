import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
import matplotlib.pyplot as plt

from ctapipe.image.toymodel import (SkewedGaussian,
                                    obtain_time_image,
                                    WaveformModel)
from ctapipe.instrument import SubarrayDescription, TelescopeDescription
from ctapipe.image.extractor import TwoPassWindowSum
from ctapipe.visualization import CameraDisplay


def test_Two_pass_window_sum_no_noise():

    class MyCameraDisplay(CameraDisplay):
        def on_pixel_clicked(self, pix_id):
            print(f"YOU CLICKED PIXEL {pix_id}")
            print(f"quantity = {self.image[pix_id]:.2f}")

    subarray = SubarrayDescription("One LST",
                tel_positions={1: np.zeros(3) * u.m},
                tel_descriptions={1: TelescopeDescription.from_name(
                    optics_name="LST", camera_name="LSTCam"
                    ),
                },
               )

    camera = subarray.tel[1].camera
    geometry = camera.geometry
    readout = camera.readout
    sampling_rate = readout.sampling_rate.to_value("GHz")
    print(sampling_rate)
    n_samples = 30  # LSTCam & NectarCam specific
    max_time_readout = (n_samples / sampling_rate) * u.ns

    # True image settings
    x = 0. * u.m
    y = 0. * u.m
    length = 0.2 * u.m
    width = 0.05 * u.m
    psi = 45.0 * u.deg
    skewness = 0.
    # we want to place the peak time of the image just about the readout window
    time_gradient = u.Quantity(max_time_readout.value / length.value, u.ns / u.m)
    time_intercept = u.Quantity(max_time_readout.value / 2, u.ns)
    intensity = 600
    nsb_level_pe = 0

    # create the image
    m = SkewedGaussian(x, y, length, width, psi, skewness)
    true_charge, true_signal, true_noise = m.generate_image(geometry,
                                                            intensity=intensity,
                                                            nsb_level_pe=nsb_level_pe)
    signal_pixels = true_signal > 0
    # create a pulse-times image
    # only signal to start
    time_noise = np.random.uniform(0, 0, geometry.n_pixels)
    time_signal = obtain_time_image(geometry.pix_x,
                                    geometry.pix_y,
                                    x,
                                    y,
                                    psi,
                                    time_gradient,
                                    time_intercept)

    true_time = np.average(
        np.column_stack([time_noise, time_signal]),
        weights=np.column_stack([true_noise, true_signal]) + 1,
        axis=1
    )

    assert np.count_nonzero(true_time) == np.count_nonzero(time_signal)
    assert np.count_nonzero(time_noise) == 0

    plt.figure()
    disp = MyCameraDisplay(geometry, true_charge, title="true_charge")
    disp.add_colorbar()
    disp.enable_pixel_picker()
    plt.figure()
    disp2 = MyCameraDisplay(geometry, true_time, title="true_time")
    disp2.add_colorbar()
    disp2.enable_pixel_picker()
    plt.show()

    # Define the model for the waveforms to fill with the information from
    # the simulated image
    waveform_model = WaveformModel.from_camera_readout(readout)
    waveforms = waveform_model.get_waveform(true_charge, true_time, n_samples)
    selected_gain_channel = np.zeros(true_charge.size, dtype=np.int64)

    # Define the extractor
    extractor = TwoPassWindowSum(subarray=subarray)

    # Select the signal pixels for which the integration window is well inside
    # the readout window (in this case we can require a more strict precision)
    true_peaks = np.rint(true_time * sampling_rate).astype(np.int64)
    true_signal_peaks = true_peaks[signal_pixels]
    print("np.min(true_signal_peaks)")
    print(np.min(true_signal_peaks))
    print("np.max(true_signal_peaks)")
    print(np.max(true_signal_peaks))

    # integration of 5 samples centered on peak + 1 sample of error
    min_good_sample = 2 + 1
    max_good_sample = n_samples - 1 - min_good_sample
    integration_window_inside = (true_peaks >= min_good_sample) & (true_peaks < max_good_sample)

    print("min peak position of nice_signal_pixels")
    print(np.min(true_peaks[signal_pixels & integration_window_inside]))
    print("max peak position of nice_signal_pixels")
    print(np.max(true_peaks[signal_pixels & integration_window_inside]))

    # Test only the 1st pass
    extractor.disable_second_pass = True
    charge_1, pulse_time_1 = extractor(waveforms, 1, selected_gain_channel)
    assert_allclose(charge_1[signal_pixels & integration_window_inside],
                    true_charge[signal_pixels & integration_window_inside], rtol=0.15)
    assert_allclose(pulse_time_1[signal_pixels & integration_window_inside],
                    true_time[signal_pixels & integration_window_inside], rtol=0.15)

    # Test also the 2nd pass
    extractor.disable_second_pass = False
    charge_2, pulse_time_2 = extractor(waveforms, 1, selected_gain_channel)

    # Check that we have gained signal charge by using the 2nd pass
    # This also checks that the test image has triggered the 2nd pass
    # (i.e. it is not too good)
    reco_charge1 = np.sum(charge_1[signal_pixels & integration_window_inside])
    reco_charge2 = np.sum(charge_2[signal_pixels & integration_window_inside])
    assert (reco_charge2 / reco_charge1) >= 1

    # Test only signal pixels for which it is expected to find most of the
    # charge well inside the readout window
    assert_allclose(charge_2[signal_pixels & integration_window_inside],
                    true_charge[signal_pixels & integration_window_inside],
                    rtol=0.1)
    assert_allclose(pulse_time_2[signal_pixels & integration_window_inside],
                    true_time[signal_pixels & integration_window_inside],
                    rtol=0.1)

    # then check again using all pixels with less precision
    # assert_allclose(charge_2[signal_pixels], true_charge[signal_pixels], rtol=0.4)
    # assert_allclose(pulse_time_2[signal_pixels], true_time[signal_pixels], rtol=0.4)
