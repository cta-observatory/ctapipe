import numpy as np
import astropy.units as u

from ctapipe.image.toymodel import WaveformModel
from ctapipe.instrument.camera.readout import CameraReadout

import numpy as np
from sklearn.cluster import DBSCAN



def make_simple_camera_readout():
    reference_pulse_shape = np.array([[0.0, 0.4, 1.0, 0.5, 0.1]])
    reference_pulse_sample_width = 1 * u.ns
    sampling_rate = 1 * u.GHz
    return CameraReadout(
        name="TestCam",
        sampling_rate=sampling_rate,
        reference_pulse_shape=reference_pulse_shape,
        reference_pulse_sample_width=reference_pulse_sample_width,
        n_channels=1,
        n_pixels=1,
        n_samples=10,
    )


def test_simple_ref_pulse_shape_from_camera_readout():
    readout = make_simple_camera_readout()
    model = WaveformModel.from_camera_readout(readout)

    assert model.n_channels == 1
    assert model.ref_interp_y.shape[0] == 1
    assert np.isclose((model.ref_interp_y.sum(-1) * model.ref_width_ns)[0], 1.0)

    waveform = model.get_waveform(np.array([5.0]), np.array([4.0]), n_samples=10)
    assert waveform.shape == (1, 1, 10)
    assert np.all(waveform >= 0)


def plot_reference_pulse():
    import matplotlib.pyplot as plt

    #readout = make_simple_camera_readout()
    model = WaveformModel(readout)

    reference_pulse_shape = readout.reference_pulse_shape
    reference_pulse_sample_width = readout.reference_pulse_sample_width
    x_raw = np.arange(reference_pulse_shape.shape[-1]) * reference_pulse_sample_width.to_value(u.ns)
    y_raw = reference_pulse_shape[0]
    x_interp = model.ref_interp_x
    y_interp = model.ref_interp_y[0]

    plt.figure()
    plt.step(x_raw, y_raw, where="mid", marker="o", label="raw reference pulse")
    plt.plot(x_interp, y_interp, label="interpolated pulse")
    plt.xlabel("time (ns)")
    plt.ylabel("amplitude")
    plt.title("Simple reference pulse shape")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return model


if __name__ == "__main__":
    #plot_reference_pulse()
    # One feature
    x = np.array([1.0, 1.1, 1.2, 1.3, 5.0, 5.1, 5.2, 10.0])
    
    # Weight of each point
    weights = np.array([1, 2, 1, 3, 1, 2, 1, 10])
    
    # DBSCAN
    db = DBSCAN(eps=0.3, min_samples=4)
    
    labels = db.fit_predict(x.reshape(-1, 1), sample_weight=weights)
    
    print(labels)
