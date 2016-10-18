"""
"""

from ctapipe.core import Container
import numpy as np


__all__ = ['RawData', 'RawCameraData', 'MCShowerData', 'MCEvent', 'MCCamera',
           'CalibratedCameraData', 'RecoShowerGeom', 'RecoEnergy', 'GammaHadronClassification']


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
        return_string = self._name + ":\n"
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
        self.add_item('tel', dict())

    def __str__(self):
        return_string = super().__str__() + "\n"
        npix = np.sum([np.sum(t.photo_electrons > 0)
                       for t in self.tel.values()])
        return_string += "total photo_electrons: {}".format(npix)
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


class RecoShowerGeom(Container):
    """
    Standard output of algorithms reconstructing shower geometry

    Parameters
    ----------

    alt : float
        reconstructed altitude
    alt_uncert : float
        reconstructed altitude uncertainty
    az : float
        reconstructed azimuth
    az_uncert : float
        reconstructed azimuth uncertainty
    core_x : float
        reconstructed x coordinate of the core position
    core_y : float
        reconstructed y coordinate of the core position
    core_uncert : float
        uncertainty of the reconstructed core position
    h_max : float
        reconstructed height of the shower maximum
    h_max_uncert : float
        uncertainty of h_max
    is_valid : bool
        direction validity flag. True if the shower direction
        was properly reconstructed by the algorithm
    tel_ids : uint array
        array containing the telescope ids used in the reconstruction
        of the shower
    average_size : float
        average size of used
    """

    def __init__(self, name='RecoShowerGeom'):
        super().__init__(name)
        self.add_item('alt')
        self.add_item('alt_uncert')
        self.add_item('az')
        self.add_item('az_uncert')
        self.add_item('core_x')
        self.add_item('core_y')
        self.add_item('core_uncert')
        self.add_item('h_max')
        self.add_item('h_max_uncert')
        self.add_item('is_valid', bool)
        self.add_item('tel_ids')
        self.add_item('average_size')
        self.add_item('goodness_of_fit')

    def __str__(self):
        return_string = self._name + ":\n"
        return_string += "altitude: {0:.2} +- {1:.2}\n".format(
            self.alt, self.alt_uncert)
        return_string += "azimuth: {0:.2} +- {1:.2}\n".format(
            self.az, self.az_uncert)
        return_string += "core: ({0:.2}, {1:.2}) +- {2:.2}\n".format(
            self.core_x, self.core_y, self.core_uncert)
        return_string += "h_max: {0:.2} +- {1:.2}\n".format(
            self.h_max, self.h_max_uncert)
        return_string += "Average size: {0:.2}\n".format(
            self.average_size)
        return_string += "Used telescopes: {}\n".format((self.tel_ids))
        return_string += "Valid reconstruction: {0}\n".format(self.is_valid)
        return_string += "Goodness of fit: {0:.2}\n".format(
            self.goodness_of_fit)
        return return_string


class RecoEnergy(Container):
    """
    Standard output of algorithms estimating energy

    Parameters
    ----------

    energy : float
        reconstructed energy
    energy_uncert : float
        reconstructed energy uncertainty
    is_valid : bool
        energy reconstruction validity flag. True if the energy
        was properly reconstructed by the algorithm
    tel_ids : uint array
        array containing the telescope ids used in the reconstruction
        of the shower
    goodness_of_fit : float
        goodness of the algorithm fit (TODO: agree on a common meaning?)
    """

    def __init__(self, name='RecoShowerGeom'):
        super().__init__(name)
        self.add_item('energy')
        self.add_item('energy_uncert')
        self.add_item('is_valid', bool)
        self.add_item('tel_ids')
        self.add_item('goodness_of_fit')

    def __str__(self):
        return_string = self._name + ":\n"
        return_string += "energy: {0:.2} +- {1:.2}\n".format(
            self.energy, self.energy_uncert)
        return_string += "Used telescopes: {0}\n".format(
            np.array2string(self.tel_ids))
        return_string += "Valid reconstruction: {0}\n".format(self.is_valid)
        return_string += "Goodness of fit: {0:.2}\n".format(
            self.goodness_of_fit)
        return return_string


class GammaHadronClassification(Container):
    """
    Standard output of gamma/hadron classification algorithms

    Parameters
    ----------

    prediction : float
        prediction of the classifier, defined between [0,1], where values
        close to 0 are more gamma-like, and values close to 1 more
        hadron-like (TODO: Do people agree on this? This is very MAGIC-like)
    is_valid : bool
        classificator validity flag. True if the predition was successful
        within the algorithm validity range
    tel_ids : uint array
        array containing the telescope ids used in the reconstruction
        of the shower
    goodness_of_fit : float
        goodness of the algorithm fit (TODO: agree on a common meaning?)
    """

    def __init__(self, name='RecoShowerGeom'):
        super().__init__(name)
        self.add_item('prediction')
        self.add_item('is_valid', bool)
        self.add_item('tel_ids')
        self.add_item('goodness_of_fit')

    def __str__(self):
        return_string = self._name + ":\n"
        return_string += "prediction: {0:.2}\n".format(self.prediction)
        return_string += "Used telescopes: {0}\n".format(
            np.array2string(self.tel_ids))
        return_string += "Valid classification: {0}\n".format(self.is_valid)
        return_string += "Goodness of fit: {0:.2}\n".format(
            self.goodness_of_fit)
        return return_string
