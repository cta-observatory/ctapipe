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

from astropy import units as u


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
    gain = event.mc.tel[telid].dc_to_pe
    calibrated = (samples - pedestal[..., None]) * gain[..., None]
    return calibrated
