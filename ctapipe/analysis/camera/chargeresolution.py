from astropy import log
import numpy as np
from math import log10, sqrt
from scipy.stats import binned_statistic


class ChargeResolution:
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
    def __init__(self, max_pe):
        """
        Parameters
        ----------
        max_pe : int
            Maximum pe to calculate the charge resolution up to.
        """
        self.max_pe = max_pe
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

    def add_charges(self, true_charge, measured_charge):
        """
        Fill the class parameters with a numpy array of true charge and
        measured (calibrated) charge from an event. The two arrays must be the
        same size.

        Parameters
        ----------
        true_charge : ndarray
            Array of true (MC) charge.
        measured_charge : ndarray
               Array of measured (calibrated) charge.
        """
        x = true_charge[(measured_charge > 0) & (true_charge > 0)]
        y = measured_charge[(measured_charge > 0) & (true_charge > 0)]
        h, _, _ = np.histogram2d(np.log10(y),
                                 np.log10(x),
                                 bins=self.variation_hist_nbins,
                                 range=self.variation_hist_range)
        self.variation_hist += h
        for trueq in np.unique(true_charge):
            if trueq == 0 or trueq > self.max_pe:
                continue
            relevant_q = measured_charge[true_charge == trueq]
            if trueq not in self.sum_dict:
                self.sum_dict[trueq] = 0
                self.n_dict[trueq] = 0
            self.sum_dict[trueq] += np.sum(np.power(relevant_q - trueq, 2))
            self.n_dict[trueq] += relevant_q.size

    def add_source(self, calibrated_source, telescopes=None):
        """
        Fill the class parameters with a calibrated source.

        Parameters
        ----------
        calibrated_source : ndarray
            `ctapipe.calib.camera.calibrators.calibrate_source` generator.
        telescopes : list
               List of telescopes to include into the charge resolution.
               If None, all telescopes will be included.
        """
        log.info('[chargeres] Adding source')
        for event in calibrated_source:
            tels = list(event.dl0.tels_with_data)
            if telescopes is not None:
                tels = []
                for tel in telescopes:
                    if tel in event.dl0.tels_with_data:
                        tels.append(tel)

            # Check source has required information
            if event.count == 0:
                # Check source is calibrated
                try:
                    if 'dl1' not in event:
                        raise KeyError
                except KeyError:
                    log.exception('[chargeres] Source has not been calibrated')
                    raise

                # Check events have true charge included
                try:
                    if np.all(event.mc.tel[tels[0]].photo_electron_image == 0):
                        raise KeyError
                except KeyError:
                    log.exception('[chargeres] Source does not '
                                  'contain true charge')
                    raise

            for telid in tels:
                true_charge = event.mc.tel[telid].photo_electron_image
                measured_charge = event.dl1.tel[telid].calibrated_image
                self.add_charges(true_charge, measured_charge)

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
        log.debug('[chargeres] Calculating charge resolution')
        true_charge = np.fromiter(iter(self.sum_dict.keys()), dtype=int)
        summed_charge = np.fromiter(iter(self.sum_dict.values()), dtype=float)
        num = np.fromiter(iter(self.n_dict.values()), dtype=int)

        chargeres = np.sqrt((summed_charge / num) + true_charge) / true_charge
        chargeres_error = chargeres * (1 / np.sqrt(2 * num))

        scale = self.goal(true_charge)
        scaled_chargeres = chargeres/scale
        scaled_chargeres_error = chargeres_error/scale

        return true_charge, chargeres, chargeres_error, \
            scaled_chargeres, scaled_chargeres_error

    def get_binned_charge_resolution(self, logarithmic_binning=True):
        """
        Calculate and obtain the charge resolution graph arrays, and bin them.

        Parameters
        ----------
        logarithmic_binning : bool
            If True, binning is performed in logarithmic space.

        Returns
        -------
        bin_true_charge : ndarray
            The X axis true charges.
        bin_chargeres : ndarray
            The Y axis charge resolution values.
        bin_chargeres_error : ndarray
            The error on the charge resolution.
        bin_scaled_chargeres : ndarray
            The Y axis charge resolution divided by the Goal.
        bin_scaled_chargeres_error : ndarray
            The error on the charge resolution divided by the Goal.
        """
        true_charge, chargeres, chargeres_error, scaled_chargeres, \
            scaled_chargeres_error = self.get_charge_resolution()

        x = true_charge
        if logarithmic_binning:
            x = np.log10(true_charge)

        def binning(array):
            return binned_statistic(x, array, 'mean', bins=60)

        def sum_errors(array):
            return np.sqrt(np.sum(np.power(array, 2)))/array.size

        def bin_errors(array):
            return binned_statistic(x, array, sum_errors, bins=60)

        bin_true_charge, _, _ = binning(true_charge)
        bin_chargeres, _, _ = binning(chargeres)
        bin_chargeres_error, _, _ = bin_errors(chargeres_error)
        bin_scaled_chargeres, _, _ = binning(scaled_chargeres)
        bin_scaled_chargeres_error, _, _ = bin_errors(scaled_chargeres_error)

        return bin_true_charge, bin_chargeres, bin_chargeres_error, \
            bin_scaled_chargeres, bin_scaled_chargeres_error

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

        requirement = ChargeResolution.limit_curves(npe, n_nsb, n_add, enf,
                                                    sigma2)
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

        goal = ChargeResolution.limit_curves(npe, n_nsb, n_add, enf, sigma2)
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
