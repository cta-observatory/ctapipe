"""Test assuring combining several steps keeps data consistent"""


def test_r1_simtel_broken_pixels(prod5_gamma_simtel_path, tmp_path):
    """Test that broken pixel and timeshift information is preserved for simtel r1"""
    from ctapipe.io import DataWriter, EventSource

    # write r1 waveforms for simtel file
    r1_path = tmp_path / "events.r1.h5"

    with EventSource(prod5_gamma_simtel_path) as source:
        with DataWriter(source, output_path=r1_path, write_r1_waveforms=True) as writer:
            for event in source:
                writer(event)

    with EventSource(r1_path) as source:
        n_checked = 0
        for event in source:
            for mon in event.monitoring.tel.values():
                assert mon.camera.coefficients.time_shift is not None
                assert mon.camera.coefficients.outlier_mask is not None
                n_checked += 1

        assert n_checked > 0
