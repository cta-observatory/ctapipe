"""
Calibration for MC (simtelarray) files.

The interpolation of the pulse shape and the adc2pe conversion in this module
are the same to their corresponding ones in
- read_hess.c
- reconstruct.c
in hessioxxx software package.

Notes
-----
Input MC version = prod2. For future MC versions the calibration
function might be different for each camera type.
"""

CALIB_SCALE = 1.05
"""
CALIB_SCALE is the factor needed to transform from mean p.e. units to units of
the single-p.e. peak: Depends on the collection efficiency, the asymmetry of
the single p.e. amplitude  distribution and the electronic noise added to the
signals. Default value is for GCT.

To correctly calibrate to number of photoelectron, a fresh SPE calibration
should be applied using a SPE sim_telarray run with an artificial light source.
"""
# TODO: add SPE calibration


def mc_r0_to_dl0_calibration(event, telid):
    """
    Perform the conversion from raw R0 data to dl0 data
    (ADC Samples -> Pedestal Subtracted ADC ->
    Scaled&Offset PE Samples (uint16) -> PE Samples)
    in one step for the MC data.

    This is required as hessio files provide r0, not dl0

    Parameters
    ----------
    event : container
        A `ctapipe` event container.
    telid : int
        Telescope ID.

    Returns
    -------
    calibrated : ndarray
        Numpy array of shape (n_chan, n_pix, n_samples) containing the samples
        calibrated to photoelectrons.

    """
    # TODO: Create r0 and r1 containers, and do conversion to dl0 at IO
    samples = event.dl0.tel[telid].adc_samples
    n_samples = samples.shape[2]
    pedestal = event.mc.tel[telid].pedestal / n_samples
    gain = event.mc.tel[telid].dc_to_pe * CALIB_SCALE
    calibrated = (samples - pedestal[..., None]) * gain[..., None]
    return calibrated
