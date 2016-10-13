"""
"""

from ctapipe.core import Container
import numpy as np


__all__ = ['EventContainer', 'RawData', 'RawCameraData', 'MCShowerData','MCEvent',
            'MCCamera', 'CalibratedCameraData']
class EventContainer(Container):
    """ Top-level container for all event information """
    def __init__(self, name="Event"):
        self.add_item("dl0", RawData())
        self.add_item("mc", MCEvent())
        self.add_item("trig", CentralTriggerData())
        self.add_item("count")

        self.meta.add_item('tel_pos', dict())
        self.meta.add_item('pixel_pos', dict())
        self.meta.add_item('optical_foclen', dict())
        self.meta.add_item('mirror_dish_area', dict())
        self.meta.add_item('mirror_numtiles', dict())
        self.meta.add_item('source', "unknown")


class RawData(Container):
    """
    Storage of a Merged Raw Data Event

    Parameters
    ----------

    run_id : int
        run number
    event_id : int
        event number
    tels_with_data : list
        list of which telescope IDs are present
    pixel_pos : dict of ndarrays by tel_id
        (deprecated)
    tel : dict of `RawCameraData` by tel_id
        dictionary of the data for each telescope
    """

    def __init__(self, name="RawData"):
        super().__init__(name)
        self.add_item('run_id')
        self.add_item('event_id')
        self.add_item('tels_with_data')
        self.add_item('tel', dict())


class MCShowerData(Container):
    def __init__(self, name='MCShowerData'):
        super().__init__(name)
        self.add_item('energy')
        self.add_item('alt')
        self.add_item('az')
        self.add_item('core_x')
        self.add_item('core_y')
        self.add_item('h_first_int')
    def __str__(self):
        return_string  = self._name+":\n"
        return_string += "energy:   {0:.2}\n".format(self.energy)
        return_string += "altitude: {0:.2}\n".format(self.alt)
        return_string += "azimuth:  {0:.2}\n".format(self.az)
        return_string += "core x:   {0:.4}\n".format(self.core_x)
        return_string += "core y:   {0:.4}"  .format(self.core_y)
        return return_string

class MCEvent(MCShowerData):
    """
    Storage of MC event data

    Parameters
    ----------

    tel : dict of `RawCameraData` by tel_id
        dictionary of the data for each telescope

    """
    def __init__(self, name='MCEvent'):
        super().__init__(name)
        self.add_item('tel',dict())
    def __str__(self):
        return_string = super().__str__()+"\n"
        npix = np.sum([np.sum(t.photo_electrons > 0) for t in self.tel.values()])
        return_string += "total photo_electrons: {}".format( npix )
        return return_string


class CentralTriggerData(Container):
    def __init__(self, name='CentralTriggerData'):
        super().__init__(name)
        self.add_item('gps_time')
        self.add_item('tels_with_trigger')


class MCCamera(Container):
    """
    Storage of mc data used for a single telescope

    Parameters
    ----------

    pe_count : dict by channel
        (masked) arrays of true (mc) pe count in each pixel (n_pixels)

    """
    def __init__(self, tel_id):
        super().__init__("CT{:03d}".format(tel_id))
        self.add_item('photo_electrons', dict())
        # Some parameters used in calibration
        self.add_item('refshapes', dict())
        self.add_item('refstep')
        self.add_item('lrefshape')
        self.add_item('time_slice')



class RawCameraData(Container):
    """
    Storage of raw data from a single telescope

    Parameters
    ----------

    adc_sums : dict by channel
        (masked) arrays of all integrated ADC data (n_pixels)
    adc_samples : dict by channel
        (masked) arrays of non-integrated ADC sample data (n_pixels, n_samples)
    num_channels : int
        number of gain channels in camera
    num_pixels : int
        number of pixels in camera
    num_samples : int
        number of samples for camera

    """
    def __init__(self, tel_id):
        super().__init__("CT{:03d}".format(tel_id))
        self.add_item('adc_sums', dict())
        self.add_item('adc_samples', dict())
        self.add_item('calibration')
        self.add_item('pedestal')
        self.add_item('num_channels')
        self.add_item('num_pixels')
        self.add_item('num_samples')


class CalibratedCameraData(Container):
    """
    Storage of calibrated (p.e.) data from a single telescope

    Parameters
    ----------

    pe_charge : dict
        ndarrays of all calibrated data (npix)
    integration_window : dict
        bool ndarrays of shape [npix][nsamples] indicating the samples used in
        the obtaining of the charge, dependant on the integration method used
    pedestal_subtracted_adc : dict
    peakpos : dict
        position of the peak as determined by the peak-finding algorithm
        for each pixel and channel
    num_channels : int
        number of gain channels in camera
    num_pixels : int
        number of pixels in camera
    calibration_parameters : dict
        the calibration parameters used to calbrate the event
    """
    def __init__(self, tel_id):
        super().__init__("CT{:03d}".format(tel_id))
        self.add_item('pe_charge')
        self.add_item('integration_window', dict())
        self.add_item('pedestal_subtracted_adc', dict())
        self.add_item('peakpos')
        self.add_item('num_channels')
        self.add_item('num_pixels')
        self.add_item('calibration_parameters', dict())


class MuonRingParameter(Container):
    """
    Storage of a Output of muon reconstruction Event

    Parameters
    ----------

    run_id : int
        run number
    event_id : int
        event number
    ring_center_x, ring_center_y, ring_radius:
        center position and radius of the fitted ring
    ring_chi2_fit:
        chi squared of the ring fit
    ring_cov_matrix:
        covariance matrix of ring parameters
    """

    def __init__(self, name="MuonRingParameter"):
        super().__init__(name)
        self.add_item('run_id')
        self.add_item('event_id')
        self.add_item('ring_center_x')
        self.add_item('ring_center_y')
        self.add_item('ring_radius')
        self.add_item('ring_chi2_fit')
        self.add_item('ring_cov_matrix')
        self.meta.add_item('ring_fit_method')
        self.meta.add_item('inputfile')

class MuonIntensityParameter(Container):
        """
        Storage of a Output of muon reconstruction Event

        Parameters
        ----------

        run_id : int
            run number
        event_id : int
            event number
        impact_parameter: float
            reconstructed impact parameter
        impact_parameter_chi2:
            chi squared impact parameter
        intensity_cov_matrix:
            Covariance matrix of impact parameters or alternatively:
            full 5x5 covariance matrix for the complete fit (ring + impact)
        impact_parameter_pos_x, impact_parameter_pos_y:
            position on the mirror of the muon impact
        COG_x, COG_y:
            center of gravity
        optical_efficiency_muon:
            optical muon efficiency from intensity fit
        ring_completeness:
            completeness of the ring
        ring_num_pixel: int
            Number of pixels composing the ring
        ring_size:
            ring size
        off_ring_size:
            size outside of the ring
        ring_width:
            ring width
        ring_time_width:
            standard deviation of the photons time arrival
        """

        def __init__(self, name="MuonIntensityParameter"):
            super().__init__(name)
            self.add_item('run_id')
            self.add_item('event_id')
            self.add_item('ring_completeness')
            self.add_item('ring_num_pixel')
            self.add_item('ring_size')
            self.add_item('off_ring_size')
            self.add_item('ring_width')
            self.add_item('ring_time_width')
            self.add_item('impact_parameter')
            self.add_item('impact_parameter_chi2')
            self.add_item('intensity_cov_matrix')
            self.add_item('impact_parameter_pos_x')
            self.add_item('impact_parameter_pos_y')
            self.add_item('COG_x')
            self.add_item('COG_y')
            self.add_item('optical_efficiency_muon')
            self.meta.add_item('intensity_fit_method')
            self.meta.add_item('inputfile')
