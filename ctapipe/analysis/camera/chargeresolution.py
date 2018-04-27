from math import log10, sqrt
from os import makedirs
from os.path import dirname, exists

import numpy as np
from scipy.stats import binned_statistic as bs
from tables import open_file

from ctapipe.core import Component
from ctapipe.core.traits import Int, Bool

__all__ = ['ChargeResolutionCalculator']


class ChargeResolutionCalculator(Component):
    """
    Class to handle the calculation of Charge Resolution.

    Attributes
    ----------
    max_pe : int
        Maximum pe to calculate the charge resolution up to.
    sum_dict : dict
        Dictionary to store the running sum for each true charge.
    n_dict : dict
        Dictionary to store the running number for each true charge.
    variation_hist_nbins : float
        Number of bins for the variation histogram.
    variation_hist_range : list
        X and Y range for the variation histogram.
    variation_hist : `np.histogram2d`
    variation_xedges : ndarray
        Edges of the X bins for the variation histogram.
    variation_yedges : ndarray
        Edges of the Y bins for the variation histogram.
    """
    max_pe = Int(2000, help='Maximum pe to calculate the charge resolution '
                            'up to').tag(config=True)
    binning = Int(60, allow_none=True,
                  help='Number of bins for the Charge Resolution. If None, '
                       'no binning is performed.').tag(config=True)
    log_bins = Bool(True, help='Bin the x axis linearly instead of '
                               'logarithmic.').tag(config=True)

    def __init__(self, config=None, tool=None, **kwargs):
        """
        Calculator of charge resolution.

        Parameters
        ----------
        config : traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            Set to None if no configuration to pass.
        tool : ctapipe.core.Tool
            Tool executable that is calling this component.
            Passes the correct logger to the component.
            Set to None if no Tool to pass.
        reductor : ctapipe.calib.camera.reductors.Reductor
            The reductor to use to reduce the waveforms in the event.
            By default no data volume reduction is applied, and the dl0 samples
            will equal the r1 samples.
        kwargs
        """
        super().__init__(config=config, parent=tool, **kwargs)

        self.sum_array = np.zeros(self.max_pe)
        self.n_array = np.zeros(self.max_pe)
        self.sum_dict = {}
        self.n_dict = {}

        self.variation_hist_nbins = log10(self.max_pe) * 50
        self.variation_hist_range = [[log10(1), log10(self.max_pe)],
                                     [log10(1), log10(self.max_pe)]]
        h, xedges, yedges = np.histogram2d([np.nan], [np.nan],
                                           bins=self.variation_hist_nbins,
                                           range=self.variation_hist_range)
        self.variation_hist = h
        self.variation_xedges = xedges
        self.variation_yedges = yedges

        self.storage_arrays = ['max_pe', 'sum_array', 'n_array',
                               'variation_hist_nbins', 'variation_hist_range',
                               'variation_hist', 'variation_xedges',
                               'variation_yedges']

    def add_charges(self, true_charge, measured_charge):
        """
        Fill the class parameters with a numpy array of true charge and
        measured (calibrated) charge from an event. The two arrays must be the
        same size.

        Parameters
        ----------
        true_charge : ndarray
            Array of true (MC) charge.
            Obtained from event.mc.tel[telid].image[channel]
        measured_charge : ndarray
            Array of measured (dl1 calibrated) charge.
            Obtained from event.mc.tel[tel_id].photo_electron_image
        """
        above_0 = (measured_charge > 0) & (true_charge > 0)
        x = true_charge[above_0]
        y = measured_charge[above_0]
        h, _, _ = np.histogram2d(np.log10(y),
                                 np.log10(x),
                                 bins=self.variation_hist_nbins,
                                 range=self.variation_hist_range)
        self.variation_hist += h

        in_range = (true_charge > 0) & (true_charge <= self.max_pe)
        true_q = true_charge[in_range]
        measured_q = measured_charge[in_range]
        np.add.at(self.sum_array, true_q - 1, np.power(measured_q - true_q, 2))
        np.add.at(self.n_array, true_q - 1, 1)

    def get_charge_resolution(self):
        """
        Calculate and obtain the charge resolution graph arrays.

        Returns
        -------
        true_charge : ndarray
            The X axis true charges.
        chargeres : ndarray
            The Y axis charge resolution values.
        chargeres_error : ndarray
            The error on the charge resolution.
        scaled_chargeres : ndarray
            The Y axis charge resolution divided by the Goal.
        scaled_chargeres_error : ndarray
            The error on the charge resolution divided by the Goal.
        """
        self.log.debug('[chargeres] Calculating charge resolution')

        n_1 = self.n_array > 0
        n = self.n_array[n_1]
        true = (np.arange(self.max_pe) + 1)[n_1]
        sum_ = self.sum_array[n_1]

        res = np.sqrt((sum_ / n) + true) / true
        res_error = res * (1 / np.sqrt(2 * n))

        scale = self.goal(true)
        scaled_res = res / scale
        scaled_res_error = res_error / scale

        if self.binning is not None:
            x = true
            if self.log_bins:
                x = np.log10(true)

            def binning(array):
                return bs(x, array, 'mean', bins=self.binning)

            def sum_errors(array):
                return np.sqrt(np.sum(np.power(array, 2))) / array.size

            def bin_errors(array):
                return bs(x, array, sum_errors, bins=self.binning)

            true, _, _ = binning(true)
            res, _, _ = binning(res)
            res_error, _, _ = bin_errors(res_error)
            scaled_res, _, _ = binning(scaled_res)
            scaled_res_error, _, _ = bin_errors(scaled_res_error)

        return true, res, res_error, scaled_res, scaled_res_error

    @staticmethod
    def limit_curves(npe, n_nsb, n_add, enf, sigma2):
        """
        Equation for calculating the Goal and Requirement curves, as defined
        in SCI-MC/121113.
        https://portal.cta-observatory.org/recordscentre/Records/SCI/
        SCI-MC/measurment_errors_system_performance_1YQCBC.pdf

        Parameters
        ----------
        npe : ndarray
            Number of photoeletrons (variable).
        n_nsb : float
            Number of NSB photons.
        n_add : float
            Number of photoelectrons from additional noise sources.
        enf : float
            Excess noise factor.
        sigma2 : float
            Percentage ofmultiplicative errors.
        """
        return (np.sqrt((n_nsb + n_add) + np.power(enf, 2) * npe +
                        np.power(sigma2 * npe, 2)) / npe).astype(float)

    @staticmethod
    def requirement(npe):
        """
        CTA requirement curve.

        Parameters
        ----------
        npe : ndarray
            Number of photoeletrons (variable).
        """
        n_nsb = sqrt(4.0 + 3.0)
        n_add = 0
        enf = 1.2
        sigma2 = 0.1
        defined_npe = 1000

        # If npe is not an array, temporarily convert it to one
        npe = np.array([npe])

        lc = ChargeResolutionCalculator.limit_curves
        requirement = lc(npe, n_nsb, n_add, enf, sigma2)
        requirement[npe > defined_npe] = np.nan

        return requirement[0]

    @staticmethod
    def goal(npe):
        """
        CTA goal curve.

        Parameters
        ----------
        npe : ndarray
            Number of photoeletrons (variable).
        """
        n_nsb = 2
        n_add = 0
        enf = 1.1152
        sigma2 = 0.05
        defined_npe = 2000

        # If npe is not an array, temporarily convert it to one
        npe = np.array([npe])

        lc = ChargeResolutionCalculator.limit_curves
        goal = lc(npe, n_nsb, n_add, enf, sigma2)
        goal[npe > defined_npe] = np.nan

        return goal[0]

    @staticmethod
    def poisson(npe):
        """
        Poisson limit curve.

        Parameters
        ----------
        npe : ndarray
            Number of photoeletrons (variable).
        """
        # If npe is not an array, temporarily convert it to one
        npe = np.array([npe])
        poisson = np.sqrt(npe) / npe

        return poisson[0]

    def save(self, path):
        output_dir = dirname(path)
        if not exists(output_dir):
            self.log.info("[output] Creating directory: {}".format(output_dir))
            makedirs(output_dir)
        self.log.info("Saving Charge Resolution file: {}".format(path))

        with open_file(path, mode="w", title="ChargeResolutionFile") as f:
            group = f.create_group("/", 'ChargeResolution', '')
            for arr in self.storage_arrays:
                f.create_array(group, arr, getattr(self, arr), arr)

    def load(self, path):
        self.log.info("Loading Charge Resolution file: {}".format(path))
        with open_file(path, mode="r") as f:
            for arr in self.storage_arrays:
                setattr(self, arr, f.get_node("/ChargeResolution", arr).read())
