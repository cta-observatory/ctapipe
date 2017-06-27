from itertools import chain
import astropy.units as u
from astropy.table import Table
from scipy.optimize import minimize
import scipy.integrate as integrate

import numpy as np

from matplotlib import pyplot as plt


__all__ = ["SensitivityPointSource"]


def crab_source_rate(energy):
    '''
    function for a pseudo-Crab point-source rate:
        dN/dE = 3e-7  * (E/TeV)**-2.5 / (TeV * m² * s)
    (watch out: unbroken power law... not really true)
    norm and spectral index reverse engineered from HESS plot...

    Parameters
    ----------
    energy : astropy quantity
        energy for which the rate is desired

    Returns
    -------
    flux : astropy quantity
        differential flux at E

    '''
    return 3e-7 * (energy/u.TeV)**-2.5 / (u.TeV * u.m**2 * u.s)


def CR_background_rate(energy):
    '''
    function for the Cosmic Ray background rate:
        dN/dE = 0.215 * (E/TeV)**-8./3 / (TeV * m² * s * sr)
    (simple power law, no knee/ankle)
    norm and spectral index reverse engineered from "random" CR plot...

    Parameters
    ----------
    energy : astropy quantity
        energy for which the rate is desired

    Returns
    -------
    flux : astropy quantity
        differential flux at E

    '''
    return 100 * 0.1**(8./3) * (energy/u.TeV)**(-8./3) / (u.TeV * u.m**2 * u.s * u.sr)


def Eminus2(energy, unit=u.TeV):
    '''
    boring, old, unnormalised E^-2 spectrum

    Parameters
    ----------
    energy : astropy quantity
        energy for which the rate is desired

    Returns
    -------
    flux : astropy quantity
        differential flux at E

    '''
    return (energy/unit)**(-2) / (unit * u.s * u.m**2)


def make_mock_event_rate(spectrum, bin_edges, e_unit=u.TeV, log_e=False, norm=None):
    """
    Creates a histogram with a given binning and fills it according to a spectral function

    Parameters
    ----------
    spectrum : function object
        function of the differential spectrum that shall be sampled into the histogram
        ought to take the energy as an astropy quantity as sole argument
    bin_edges : numpy array, optional (default: None)
        bin edges of the histogram that is to be filled
    e_unit : astropy unit (default: u.GeV)
        unit of the histogram's axis
    log_e : bool, optional (default: False)
        tell if the values in `bin_edges` are given in logarithm
    norm : float, optional (default: None)
        normalisation factor for the histogram that's being filled
        sum of all elements in the array will be equal to `norm`

    Returns
    -------
    rates : numpy array
        histogram of the (non-energy-differential) event rate of the proposed spectrum
    """

    def spectrum_value(e):
        """
        `scipy.integrate` does not like units during integration. use this as a quick fix
        """
        return spectrum(e).value

    rates = []
    if log_e:
        bin_edges = 10**bin_edges
    for l_edge, h_edge in zip(bin_edges[:-1], bin_edges[1:]):
        bin_events = integrate.quad(spectrum_value, l_edge.value, h_edge.value)[0]
        rates.append(bin_events)

    # units have been strip for the integration. the unit of the result is the unit of the
    # function: spectrum(e) times the unit of the integrant: e -- for the latter use the
    # first entry in `bin_edges`
    rates = np.array(rates)*spectrum(bin_edges[0]).unit * bin_edges[0].unit

    # if `norm` is given renormalise the sum of the `rates`-bins to this value
    if norm:
        rates *= norm/np.sum(rates)

    return rates


def sigma_lima(Non, Noff, alpha=0.2):
    """
    Compute the significance according to Eq. (17) of Li & Ma (1983).

    Parameters
    ----------
    Non : integer/float
        Number of on counts
    Noff : integer/float
        Number of off counts
    alpha : float, optional (default: 0.2)
        Ratio of on-to-off exposure

    Returns
    -------
    sigma : float
        the significance of the given off and on counts
    """

    alpha1 = alpha + 1.0
    sum = Non + Noff
    arg1 = Non / sum
    arg2 = Noff / sum
    term1 = Non
    term2 = Noff
    if Non > 0:
        term1 *= np.log((alpha1/alpha)*arg1)
    if Noff > 0:
        term2 *= np.log(alpha1*arg2)
    sigma = np.sqrt(2.0 * (term1 + term2))

    return sigma


def diff_to_X_sigma(scale, N, alpha, X=5):
    """
    calculates the significance according to `sigma_lima` and returns the squared
    difference to X. To be used in a minimiser that determines the necessary source
    intensity for a detection of given significance of `X` sigma
    The square here is only to have a continuously differentiable function with a smooth
    turning point at the minimum -- in contrast to an absolute-function that makes a
    sharp turn at the minimum.

    Parameters
    ----------
    scale : python list with a single float
        this is the variable in the minimisation procedure
        it scales the number of gamma events
    N : shape (2) list
        the signal count in the on- (index 0)
        and the background count int the off-region (index 1)
        the events in the on-region are to be scaled by `scale[0]`
        the background rate in the on-region is estimated as `alpha` times off-count
    alpha : float
        the ratio of the on and off areas
    X : float, optional (default: 5)
        target significance in multiples of "sigma"

    Returns
    -------
    (sigma-X)**2 : float
        squared difference of the significance to `X`
        minimise this function for the number of gamma events needed for your desired
        significance of `X` sigma
    """

    Non = N[1]*alpha + N[0]*scale[0]
    Noff = N[1]
    sigma = sigma_lima(Non, Noff, alpha)
    return (sigma-X)**2


#   ######  ##          ###     ######   ######
#  ##    ## ##         ## ##   ##    ## ##    ##
#  ##       ##        ##   ##  ##       ##
#  ##       ##       ##     ##  ######   ######
#  ##       ##       #########       ##       ##
#  ##    ## ##       ##     ## ##    ## ##    ##
#   ######  ######## ##     ##  ######   ######
class SensitivityPointSource:
    """
    class to calculate effective areas and the sensitivity to a known point-source
    TODO extend sensitvity module
        • add extended source?
        • add pseudo experiment for low exposure times?
        ✔ add methods for time stamp generation and energy sampling of events

    Parameters
    ----------
    reco_energies: dictionary of quantified numpy arrays, optional
        lists of reconstructed of the selected events for each channel
    mc_energies: dictionary of quantified numpy arrays, optional
        lists of simulated energies of the selected events for each channel
    energy_bin_edges : dictionary of numpy arrays, optional
        lists of the bin edges for the various histograms for each channel;
        assumes binning in log10(energy)
        you should provide this at some point if not done here
    energy_unit : astropy quantity, optional (default: u.TeV)
        your favourite energy unit
    flux_unit : astropy quantity, optional (default: 1 / (u.TeV * u.m**2 * u.s))
        your favourite differential flux unit
    """

    def __init__(self, reco_energies=None, mc_energies=None, energy_bin_edges=None,
                 energy_unit=u.TeV, flux_unit=1/(u.TeV*u.m**2*u.s)):

        self.reco_energies = reco_energies
        self.mc_energies = mc_energies
        self.energy_bin_edges = energy_bin_edges

        try:
            self.class_list = reco_energies.keys()
        except AttributeError:
            self.class_list = mc_energies.keys()

        self.energy_unit = energy_unit
        self.flux_unit = flux_unit

    def get_effective_areas(self, generator_areas,
                            n_simulated_events=None,
                            generator_spectra=None,
                            generator_energy_hists={}
                            ):
        """
        calculates the effective areas for the provided channels and stores them in the
        class instance

        Parameters
        ----------
        generator_areas : astropy quantities
            the area for each channel within which the shower impact position was
            generated
        n_simulated_events : dictionary of integers, optional (defaults: None)
            number of events used in the MC simulation for each channel
        generator_spectra : dictionary of functors, optional (default: None)
            function object for the differential generator flux of each channel
        generator_energy_hists : numpy arrays, optional (default: {})
            energy histogram of the generated events for each channel binned according to
            `.energy_bin_edges`

        Returns
        -------
        eff_area_gam, eff_area_pro : numpy arrays
            histograms of the effective areas of gammas and protons binned according to
            `.bin_edges_gam` and `.bin_edges_pro`

        Notes
        -----
        either give the histogram of the energy distributions at MC generator level with
        `generator_energy_hists` or create them on the fly with `n_simulated_events` and
        `spectra`
        """

        if (n_simulated_events is not None and generator_spectra is not None) == \
                (generator_energy_hists):
                    raise ValueError("use either (n_simulated_events and generator"
                                     "_spectra) or generator_energy_hists to set the MC "
                                     "generated energy spectrum -- not both")

        if not generator_energy_hists:
            # generate the histograms for the energy distributions of the Monte Carlo
            # generated events given the generator spectra and the number of generated
            # events
            for cl in self.class_list:
                generator_energy_hists[cl] = make_mock_event_rate(
                                    generator_spectra[cl], norm=n_simulated_events[cl],
                                    bin_edges=self.energy_bin_edges[cl], log_e=False)
        self.generator_energy_hists = generator_energy_hists

        # an energy-binned histogram of the effective areas
        # binning according to .energy_bin_edges[cl]
        self.effective_areas = {}

        # an energy-binned histogram of the selected events
        # binning according to .energy_bin_edges[cl]
        self.selected_events = {}

        # generate the histograms for the energy distributions of the selected events
        for cl in self.class_list:
            self.selected_events[cl] = np.histogram(self.mc_energies[cl],
                                                    bins=self.energy_bin_edges[cl])[0]

            # the effective areas are the selection efficiencies per energy bin multiplied
            # by the area in which the Monte Carlo events have been generated in
            efficiency = self.selected_events[cl] / self.generator_energy_hists[cl]
            self.effective_areas[cl] = efficiency * generator_areas[cl]

        return self.effective_areas

    def generate_event_weights_histogram(self, spectra={'g': crab_source_rate,
                                                        'p': CR_background_rate},
                                         extensions={'p': 6*u.deg},
                                         observation_time=50*u.h):
        """
        given a source rate and the effective area, calculates the number of expected
        events within a given observation time
        then, produces weights to scale the selected events according to source/background
        spectrum and observation time

        Parameters
        ----------
        spectra : dictionary of functors, optional (default: 'g': `crab_source_rate`,
                                                             'p': `CR_background_rate`)
            functions for the differential source and background rates
        extensions : dictionary of astropy quantities, optional (default: {'p': 6*u.deg})
            opening angle of the view-cone the events have been generated in, if any
            don't set the key if a channel was generated as a point-source
            note: if you use an extension, the flux needs to accomodate that as well
        observation_time : astropy quantity, optional (default: 50*u.h)
            length of the assumed exposure

        Returns
        -------
        weights : dictionary of numpy arrays
            weights for the selected events so that they are scaled to the number of
            expected events in `exp_events_per_energy_bin` for every energy bin
        """

        # an energy-binned histogram of the number of events expected from the given flux
        # binning according to .energy_bin_edges[cl]
        self.exp_events_per_energy_bin = {}
        for cl in self.class_list:
            event_rate = make_mock_event_rate(spectra[cl], log_e=False,
                                              bin_edges=self.energy_bin_edges[cl])
            if cl in extensions:
                omega = 2*np.pi*(1 - np.cos(extensions[cl]))*u.rad**2
                event_rate *= omega

            self.exp_events_per_energy_bin[cl] = (event_rate * observation_time *
                                                  self.effective_areas[cl]).si

        self.event_weights = {}
        for cl in self.class_list:
            weights = (self.exp_events_per_energy_bin[cl] / self.selected_events[cl]).si
            self.event_weights[cl] = weights[np.clip(
                                np.digitize(self.mc_energies[cl],
                                            self.energy_bin_edges[cl])-1,
                                0, len(self.energy_bin_edges[cl])-2)]

            self.event_weights[cl] = np.array(self.event_weights[cl])

        return self.event_weights

    def generate_event_weights(self, n_simulated_events, e_min_max, spectra,
                               generator_areas,
                               extensions={'p': 6*u.deg},
                               observation_time=50*u.h,
                               generator_gamma={"g": 2, "p": 2}):

        """
        generates a weight for every event

        Parameters
        ----------
        n_simulated_events : dictionary
            total number of simulated events for every channel
        e_min_max : dictionary of tuples
            lower and upper energy limits used in the Monte Carlo event generator
        generator_areas : dictionary
            area within which the shower impact point has been distributed by the
            Monte Carlo event generator
        extensions : dictionary, optional (default: {'p': 6*u.deg})


        """
        self.event_weights = {}
        self.exp_events_per_energy_bin = {}
        for cl in self.class_list:
            # event weights for a flat energy distribution
            e_w = (self.mc_energies[cl])**generator_gamma[cl] \
                  * (e_min_max[cl][1]**(1-generator_gamma[cl]) -
                     e_min_max[cl][0]**(1-generator_gamma[cl])) / \
                    (1-generator_gamma[cl]) \
                  * generator_areas[cl] \
                  * (1 if (cl not in extensions) else
                     2*np.pi*(1-np.cos(extensions[cl]))*u.rad**2) \
                  * observation_time \
                  / n_simulated_events[cl]

            # multiply these flat event weights by the flux to get weights corresponding
            # to the number of expected events from that flux
            # the event weight should be unit-less; the call to `dimensionless_unscaled`
            # is an explicit test of this requirement and resolves any possible mixed
            # units to unity
            # QUESTION use `.si` to resolve mixed units to allow unified weights?
            # e.g. `observation_time` could be set to `1` -- without unit -- to get
            # `event_weights` in units of 1/time and be later multiplied by different
            # observation times
            self.event_weights[cl] = \
                (e_w * spectra[cl](self.mc_energies[cl])).to(u.dimensionless_unscaled)

            # now, for the fun of it, make an energy-binned histogram of the events
            self.exp_events_per_energy_bin[cl], _ = \
                np.histogram(self.mc_energies[cl],
                             bins=self.energy_bin_edges[cl],
                             weights=self.event_weights[cl])

        return self.event_weights

    def get_sensitivity(self, sensitivity_energy_bin_edges,
                        alpha, signal_list=("g"), mode="MC",
                        sensitivity_source_flux=crab_source_rate,
                        min_n=10, max_background_ratio=.05):
        """
        finally calculates the sensitivity to a point-source
        TODO still need to implement statistical error on the sensitivity

        Parameters
        ----------
        alpha : float
            area-ratio of the on- over the off-region
        sensitivity_energy_bin_edges : numpy array
            array of the bin edges for the sensitivity calculation
        mode : string ["MC", "Data"] (default: "MC")
            interprete the signal/not-signal channels in all the dictionaries as
            gamma/background ("MC") or as on-region/off-region ("Data")
            - if "MC":
                the signal channel is taken as the part comming from the source and
                the background channels multiplied by `alpha` is used as the background
                part in the on-region; the background channels themselves are taken as
                coming from the off-regions
            - if "Data":
                the signal channel is taken as the counts reconstructed in the on-region
                the counts from the background channels multiplied by `alpha` are taken as
                the background estimate for the on-region
        min_n : integer, optional (default: 10)
            minimum number of events per energy bin -- if the number is smaller, scale up
            all events to sum up to this
        max_background_ratio : float, optional (default: 0.05)
            maximal background contamination per bin -- if fraction of protons in a bin
            is larger than this, scale up the gammas events accordingly
        sensitivity_source_flux : callable, optional (default: `crab_source_rate`)
            function of the flux the sensitivity is calculated with

        Returns
        -------
        sensitivities : astropy.table.Table
            the sensitivity for every energy bin of `sensitivity_energy_bin_edges`
        """

        # sensitivities go in here
        sensitivities = Table(names=("Energy", "Sensitivity", "Sensitivity_base"))
        try:
            sensitivities["Energy"].unit = sensitivity_energy_bin_edges.unit
        except AttributeError:
            sensitivities["Energy"].unit = self.energy_unit
        sensitivities["Sensitivity"].unit = self.flux_unit
        sensitivities["Sensitivity_base"].unit = self.flux_unit

        if hasattr(self, "event_weights"):
            # in case we do have event weights, we sum them within the energy bin
            # count also the square of the weights for the error estimator
            count_events = lambda mask: np.sum(self.event_weights[cl][mask])
            count_square = lambda mask: np.sum(self.event_weights[cl][mask]**2)
        else:
            # otherwise we simply check the length of the masked energy array
            # since the weights are 1 here, `count_square` is the same as `count_events`
            count_events = lambda mask: len(self.reco_energies[cl][mask])
            count_square = count_events

        # loop over all energy bins
        # the bins are spaced logarithmically: use the geometric mean as the bin-centre,
        # so when plotted logarithmically, they appear at the middle between the bin-edges
        for elow, ehigh, emid in zip(sensitivity_energy_bin_edges[:-1],
                                     sensitivity_energy_bin_edges[1:],
                                     np.sqrt(sensitivity_energy_bin_edges[:-1] *
                                             sensitivity_energy_bin_edges[1:])):

            N_events = np.zeros(2)  # [on-signal, off-background]
            variance = np.zeros(2)  # [on-signal, off-background]

            # count the (weights of the) events in the on and off regions for this
            # energy bin
            for cl in self.class_list:
                # single out the events in this energy bin
                e_mask = (self.reco_energies[cl] > elow) & \
                         (self.reco_energies[cl] < ehigh)

                # count the events as the sum of their weights within this energy bin
                if cl in signal_list:
                    N_events[0] += count_events(e_mask)
                    variance[0] += count_square(e_mask)
                else:
                    N_events[1] += count_events(e_mask)
                    variance[1] += count_square(e_mask)

            if mode.lower() == "data":
                # the background estimate for the on-region is `alpha` times the
                # background in the off-region
                # if running on data, the signal estimate for the on-region is the counts
                # in the on-region minus the background estimate for the on-region
                N_events[0] -= N_events[1]*alpha
                variance[0] -= variance[1]*alpha

            # If we have no counts in the on-region, there is no sensitivity.
            # If on data the background estimate from the off-region is larger than the
            # counts in the on-region, `sigma_lima` will break! Skip those cases, too.
            if N_events[0] <= 0:
                continue

            # find the scaling factor for the gamma events that gives a 5 sigma discovery
            # in this energy bin
            scale = minimize(diff_to_X_sigma, [1e-3],
                             args=(N_events, alpha),
                             method='L-BFGS-B', bounds=[(1e-4, None)],
                             options={'disp': False}
                             ).x[0]

            scale_base = scale

            # scale up the gamma events by this factor
            N_events[0] *= scale

            # check if there are sufficient events in this energy bin
            scale *= check_min_n(N_events, min_n=min_n, alpha=alpha)

            # check if the relative amount of protons in this bin is sufficiently small
            scale *= check_background_contamination(
                    N_events, alpha=alpha, max_background_ratio=max_background_ratio)

            # get the flux at the bin centre
            flux = sensitivity_source_flux(emid).to(self.flux_unit)

            # and scale it up by the determined factor
            sensitivity = flux*scale
            sensitivity_base = flux*scale_base

            # store results in table
            sensitivities.add_row([emid, sensitivity, sensitivity_base])

        return sensitivities

    def calculate_sensitivities(self,
                                # arguments for `get_effective_areas`
                                generator_areas,
                                n_simulated_events=None,
                                generator_spectra=None,
                                generator_energy_hists={},

                                # arguments for `generate_event_weights`
                                extensions={'p': 6*u.deg},
                                observation_time=50*u.h,
                                generator_gamma=None,
                                e_min_max=None,
                                spectra=None,

                                # arguments for `get_sensitivity`
                                reco_energies=None,
                                sensitivity_energy_bin_edges=np.logspace(-1, 3, 17)*u.TeV,
                                alpha=1, min_n=10, max_background_ratio=.05,
                                ):
        """
        wrapper that calls all functions to calculate the point-source sensitivity

        Parameters
        ----------
        cf. corresponding functions:
        `.get_effective_areas`
        `.generate_event_weights`
        `.get_sensitivity`


        Returns
        -------
        sensitivities : `astropy.table.Table`
            the sensitivity for every energy bin of `.bin_edges_gam`
        """

        # this is not needed for the sensitivity, but effective areas are nice to have so,
        # do it anyway here
        self.get_effective_areas(n_simulated_events=n_simulated_events,
                                 generator_spectra=generator_spectra,
                                 generator_energy_hists=generator_energy_hists,
                                 generator_areas=generator_areas)

        self.generate_event_weights(n_simulated_events=n_simulated_events,
                                    observation_time=observation_time,
                                    extensions=extensions,
                                    e_min_max=e_min_max,
                                    spectra=spectra,
                                    generator_gamma=generator_gamma,
                                    generator_areas=generator_areas)

        return self.get_sensitivity(
                        alpha=alpha, min_n=min_n,
                        max_background_ratio=max_background_ratio,
                        sensitivity_energy_bin_edges=sensitivity_energy_bin_edges)

    @staticmethod
    def generate_toy_timestamps(light_curves, time_window):
        """
        randomly draw time stamps within a time window
        either uniformly distributed or by following the distribution of a light curve

        Parameters
        ----------
        light_curve : dictionary of numpy arrays or floats
            light curve you want to sample time stamps from
            sum of the elements (cast to `int`) will be taken as number of draws
            if value is float and not an array, assume take than uniform draws
        time_window : tuple of floats
            min and max of the drawn time stamps

        Returns
        -------
        time_stamps : dictionary of numpy arrays
            lists of randomly drawn time stamps
        """

        time_stamps = {}
        for cl, f in light_curves.items():
            if hasattr(f, "__iter__"):
                # if `f` is any kind of iterable container, create cumulative
                # distribution, draw  randomly from it, and sample time stamps within the
                # `time_window`  accordingly
                cum_sum = np.cumsum(f)
                random_draws = np.random.uniform(0, cum_sum[-1], int(cum_sum[-1]))
                indices = np.digitize(random_draws, cum_sum)
                time_stamps[cl] = time_window[0] + indices * (time_window[1] -
                                                              time_window[0]) / len(f)
            else:
                # if `f` is just a number, draw uniformly that many events within the
                # `time_window`
                time_stamps[cl] = np.random.uniform(*time_window, f)

        return time_stamps

    @staticmethod
    def draw_events_from_flux_histogram(mc_energies, spectrum_hists, energy_bin_edges):

        drawn_indices = {}
        for cl, h in spectrum_hists.items():
            cum_sum = np.insert(np.cumsum(h), 0, 0)

            # rounding errors can make the sum slightly smaller than intended
            # add a small epsilon to push above that number
            random_draws = np.random.uniform(0, cum_sum[-1], int(cum_sum[-1]+1e-5))

            cumsum_indices = np.digitize(random_draws, cum_sum)

            cum_sums_up = cum_sum[cumsum_indices]
            cum_sums_lo = cum_sum[cumsum_indices-1]

            energies_up = energy_bin_edges[cl][cumsum_indices]
            energies_lo = energy_bin_edges[cl][cumsum_indices-1]

            drawn_energies = energies_lo + (random_draws-cum_sums_lo) * \
                (energies_up-energies_lo) / (cum_sums_up-cum_sums_lo)

            # here be numpy magic:
            # `drawn_energies[:, np.newaxis]` enables the broadcast to `mc_energies`
            #
            # `np.abs(mc_energies[cl] - drawn_energies[:, np.newaxis])` is a matrix
            # containing the absolute difference between every drawn energy with every MC
            # energy
            #
            # `np.argmin(X, axis=1)` returns the indices along the `mc_energies` axis of
            # the elements with the smallest difference, i.e. these are the MC events with
            # the energy closest to the randomly drawn energies
            drawn_indices[cl] = np.argmin(np.abs(mc_energies[cl] -
                                                 drawn_energies[:, np.newaxis]),
                                          axis=1)
        return drawn_indices

    @staticmethod
    def draw_events_from_flux_weight(mc_energies, energy_weights, N_draws):

        drawn_indices = {}
        for cl, weights in energy_weights.items():
            drawn_indices[cl] = np.random.choice(range(len(mc_energies[cl])),
                                                 size=N_draws[cl],
                                                 p=weights)
        return drawn_indices


def check_min_n(n, alpha=1, min_n=10):
    """
    check if there are sufficenly many events in this energy bin and calculates scaling
    parameter if not.
    scales `N_signal` in place.

    Parameters
    ----------
    n = [n_signal, n_backgr] : list
        the signal in the on-region (index 0) and background in the off-region (index 1)
    alpha : float (default: 1)
        ratio of the on- to off-region -- `n[1]` times `alpha` is used as background
        estimate for the on-region
    min_n : integer (default: 10)
        minimum number of desired events; if too low, scale up to this number

    Returns
    -------
    scale_a : float
        factor to scale up gamma events if insuficently many events present
    """

    n_signal, n_backgr = n[0], n[1]*alpha

    if n_signal + n_backgr < min_n:
        scale_a = (min_n-n_backgr) / n_signal
        n[0] *= scale_a
        return scale_a
    else:
        return 1


def check_background_contamination(n, alpha=1, max_background_ratio=.05):
    """
    check if there are sufficenly few proton events in this energy bin and calculates
    scaling parameter if not
    scales `N_signal` in place.

    Parameters
    ----------
    n = [n_signal, n_backgr] : list
        the signal in the on-region (index 0) and background in the off-region (index 1)
    alpha : float (default: 1)
        ratio of the on- to off-region -- `n[1]` times `alpha` is used as background
        estimate for the on-region
    max_background_ratio : float
        maximum allowed proton contamination

    Returns
    -------
    scale_r : float
        factor to scale up gamma events if too high proton contamination
    """

    n_signal, n_backgr = n[0], n[1]*alpha

    n_tot = n_signal + n_backgr
    if n_backgr / n_tot > max_background_ratio:
        scale_r = (1/max_background_ratio - 1) * n_backgr / n_signal
        n[0] *= scale_r
        return scale_r
    else:
        return 1
