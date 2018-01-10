from ctapipe.core import Container, Field, Map
from ctapipe.io.containers import ReconstructedContainer, \
    CentralTriggerContainer, InstrumentContainer, MCEventContainer, \
    R1Container, DL0Container, DL1Container, TelescopePointingContainer


class TargetioR0CameraContainer(Container):
    """
    Storage of raw data from a single telescope for Targetio reader

    Includes some extra camera-specific fields that are used in
    the R1 calibration for TargetIO files.
    """
    adc_sums = Field(None, ("numpy array containing integrated ADC data "
                            "(n_channels x n_pixels)"))
    adc_samples = Field(None, ("numpy array containing ADC samples"
                               "(n_channels x n_pixels, n_samples)"))
    first_cell_ids = Field(None, ("numpy array of the first_cell_id of each"
                                  "waveform in the camera image"))
    blockphase = Field(None, ("numpy array of the blockphase of each "
                              "waveform in the camera image"))
    row = Field(None, ("numpy array of the row of each "
                       "waveform in the camera image"))
    column = Field(None, ("numpy array of the column of each "
                          "waveform in the camera image"))
    num_samples = Field(None, "number of time samples for telescope")


class TargetioR0Container(Container):
    """
    Storage of a Merged Raw Data Event for Targetio reader
    """

    run_id = Field(-1, "run id number")
    event_id = Field(-1, "waveforms id number")
    tels_with_data = Field([], "list of telescopes with data")
    tel = Field(Map(TargetioR0CameraContainer), "map of tel_id to "
                                                "TargetioR0CameraContainer")


class TargetioDataContainer(Container):
    """
    Top-level container for all waveforms information for Targetio reader
    """

    r0 = Field(TargetioR0Container(), "Raw Data")
    r1 = Field(R1Container(), "R1 Calibrated Data")
    dl0 = Field(DL0Container(), "DL0 Data Volume Reduced Data")
    dl1 = Field(DL1Container(), "DL1 Calibrated image")
    dl2 = Field(ReconstructedContainer(), "Reconstructed Shower Information")
    mc = Field(MCEventContainer(), "Monte-Carlo data")
    trig = Field(CentralTriggerContainer(), "central trigger information")
    count = Field(0, "number of events processed")
    inst = Field(InstrumentContainer(), "instrumental information (deprecated")
    pointing = Field(Map(TelescopePointingContainer),
                     'Telescope pointing positions')
