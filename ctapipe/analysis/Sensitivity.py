from itertools import chain
import astropy.units as u
from astropy.table import Table
from scipy.optimize import minimize

import numpy as np

# tau, the superior circle number
np.tau = 2*np.pi

__all__ = ["Sensitivity_PointSource"]


def convert_astropy_array(arr, unit=None):
    """ converts a python list of quantities into a quantified numpy array in the SI unit
    of the same dimension

    Parameters
    ----------
    arr : python list
        list of quantities of same dimension (not strictly exact same unit)
    unit : astropy unit, optional (default: None)
        if set, uses this as the unit of the numpy array; if not, uses the unit of the
        first element
        ought to be of same dimension of the quantities in the list (there is no test)

    Returns
    -------
    a : quantified numpy array
    """

    if unit is None:
        unit = arr[0].unit
    return np.array([a.to(unit).value for a in arr])*unit


def crab_source_rate(energy):
    ''' function for a pseudo-Crab point-source rate
    Crab source rate:   dN/dE = 3e-7  * (E/TeV)**-2.48 / (TeV * m² * s)
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
    return 3e-7 * (energy/u.TeV)**-2.48 / (u.TeV * u.m**2 * u.s)


def CR_background_rate(energy):
    ''' function of the cosmic ray spectrum (simple power law, no knee/ankle)
    Cosmic Ray background rate: dN/dE = 0.215 * (E/TeV)**-8./3 / (TeV * m² * s * sr)
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


def Eminus2(energy, unit=u.GeV):
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


def make_mock_event_rate(spectrum, bin_edges, e_unit=u.GeV, log_e=True, norm=None):
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
    log_e : bool, optional (default: None)
        tell if the values in `bin_edges` are given in logarithm
    norm : float, optional (default: None)
        normalisation factor for the histogram that's being filled

    Returns
    -------
    rates : numpy array
        histogram of the (non-energy-differential) event rate of the proposed spectrum
    """

    rates = []
    for l_edge, h_edge in zip(bin_edges[:-1], bin_edges[1:]):
        if log_e:
            bin_centre = 10**((l_edge+h_edge)/2.) * e_unit
            bin_width = (10**h_edge-10**l_edge) * e_unit

        else:
            bin_centre = (l_edge+h_edge) * e_unit / 2.
            bin_width = (h_edge-l_edge) * e_unit
        bin_events = spectrum(bin_centre) * bin_width
        rates.append(bin_events)

    # rates is now a python list of astropy quantities
    # convert it into a quantified numpy array
    rates = convert_astropy_array(rates)

    if norm:
        rates *= norm/np.sum(rates)

    return rates


def sigma_lima(Non, Noff, alpha=0.2):
    """
    Compute the significance according to Eq. (17) of Li & Ma (1983).

    Parameters
    ----------
    Non : integer
        Number of on counts
    Noff : integer
        Number of off counts
    alpha : float, optional (default: 0.2)
        Ratio of on-to-off exposure

    Returns
    -------
    sigma : float
        the significance of the given off and on counts
    """

    alpha1 = alpha + 1.0
    sum    = Non + Noff
    arg1   = Non / sum
    arg2   = Noff / sum
    term1  = Non  * np.log((alpha1/alpha)*arg1)
    term2  = Noff * np.log(alpha1*arg2)
    sigma  = np.sqrt(2.0 * (term1 + term2))

    return sigma


def diff_to_X_sigma(scale, N_signal, N_backgr, alpha, X=5):
    """
    calculates the significance according to `sigma_lima` and returns the squared
    difference to X. To be used in a minimiser that determines the necessary source
    intensity for a detection of given significance of `X` sigma

    Parameters
    ----------
    scale : python list with a single float
        this is the variable in the minimisation procedure
        it scales the number of gamma events
    N_signal, N_backgr : shape (2) numpy arrays
        the on (index 0) and off (index 1) counts for gamma and proton events
        the gamma events are to be scaled by `scale[0]`
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

    Non = N_backgr[0] + N_signal[0] * scale[0]
    Noff = N_backgr[1] + N_signal[1] * scale[0]
    sigma = sigma_lima(Non, Noff, alpha)
    return (sigma-X)**2


class Sensitivity_PointSource():
    """
    class to calculate the sensitivity to a known point-source
    TODO:
        • add extended source?
        • add pseudo experiment for low exposure times?
    """
    def __init__(self, mc_energies, energy_bin_edges,
                 off_angles=None,
                 source_origin=None, event_origins=None,
                 energy_unit=u.GeV, flux_unit=u.GeV / (u.m**2*u.s)):
        """
        constructor, simply sets some initial parameters

        Parameters
        ----------
        mc_energies: dictionary of quantified numpy arrays
            lists of simulated energies of the selected events for each channel
        energy_bin_edges : dictionary of numpy arrays
            lists of the bin edges for the various histograms for each channel;
            assumes binning in log10(energy)
        off_angles : dictionary of numpy arrays
            lists of offset angles between the reconstructed direction and
            the point-source direction for the events of each channel
        source_origin : `astropy.coordinates.SkyCoord`, optional (default: None)
            position in the sky of the assumed point-source
        event_origins : dictionary of lists of `astropy.coordinates.SkyCoord`,
                        optional (default: None)
            lists of origins (*not* directions) of the reconstructed events
            for each channel
        energy_unit : astropy quantity, optional (default: u.GeV)
            your favourite energy unit
        flux_unit : astropy quantity, optional (default: u.GeV / (u.m**2 * u.s))
            your favourite differential flux unit

        Notes
        -----
        use either `off_angles` directly, or let the constructor compute them itself
        by using `source_origin` and `event_origins`
        """

        assert (off_angles is not None) != (event_origins is not None), \
            "use only 'off_angles' OR 'event_origins'"

        if off_angles is not None:
            self.off_angles = off_angles
        else:
            self.off_angles = {}
            for cl in event_origins.keys():
                self.off_angles[cl] = source_origin.separation(event_origins[cl])

        self.mc_energies = mc_energies
        self.energy_bin_edges = energy_bin_edges

        self.class_list = mc_energies.keys()

        self.energy_unit = energy_unit
        self.flux_unit = flux_unit

    def get_effective_areas(self, n_simulated_events=None,
                            spectra=None,
                            generator_energy_hists={},
                            generator_areas=None):
        """
        calculates the effective areas for gammas and protons and stores them in the
        class instance

        Parameters
        ----------
        n_simulated_events : dictionary of integers, optional (defaults: None)
            number of events used in the MC simulation for each channel
        spectra : dictionary of functors, optional (default: None)
            function object for the differential generator flux of each channel
        generator_energy_hists : numpy arrays, optional (default: {})
            energy histogram of the generated events for each channel binned according to
            `.energy_bin_edges`
        generator_areas : astropy quantities
            the area for each channel within which the shower impact position was
            generated

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

        if not generator_energy_hists:
            for cl in self.class_list:
                generator_energy_hists[cl] = make_mock_event_rate(
                                    spectra[cl], norm=n_simulated_events[cl],
                                    bin_edges=self.energy_bin_edges[cl])

        self.effective_areas = {}
        self.selected_events = {}

        for cl in self.class_list:
            self.selected_events[cl] = np.histogram(np.log10(self.mc_energies[cl]),
                                                    bins=self.energy_bin_edges[cl])[0]

            efficiency = self.selected_events[cl] / generator_energy_hists[cl]
            self.effective_areas[cl] = efficiency * generator_areas[cl]

        return self.effective_areas

    def get_expected_events(self, rates={'g': Eminus2, 'p': CR_background_rate},
                            extensions={'p': 6*u.deg},
                            observation_time=50*u.h):
        """
        given a source rate and the effective area, calculates the number of expected
        events within a given observation time

        Parameters
        ----------
        rates : dictionary of functors, optional (default: 'g': `Eminus2`,
            'p': `CR_background_rate`)
            functions for the differential source and background rates
        extensions : dictionary of astropy quantities, optional (defaults: {'g': None,
            'p': 6*u.deg)
            opening angle of the view-cone the events have been generated in
            put `None` for point source
            note: if you use an extension, the flux needs to accomodate that as well
        observation_time : astropy quantity, optional (default: 50*u.h)
            length of the assumed exposure
        """

        # for book keeping
        self.observation_time = observation_time

        self.exp_events_per_energy_bin = {}
        for cl in self.class_list:
            event_rate = make_mock_event_rate(rates[cl],
                                              bin_edges=self.energy_bin_edges[cl])[0]

            if cl in extensions:
                omega = np.tau*(1 - np.cos(extensions[cl]))*u.rad**2
                event_rate *= omega

            self.exp_events_per_energy_bin[cl] = event_rate * observation_time * \
                                                 self.effective_areas[cl]

        return self.exp_events_per_energy_bin

    def scale_events_to_expected_events(self):
        """
        produces weights to scale the selected events according to source/background
        spectrum and observation time

        Returns
        -------
        weights : dictionary of numpy arrays
            weights for the selected events so that they are scaled to the number of
            expected events in `exp_events_per_energy` for every energy bin
        """

        self.event_weights = {}
        for cl in self.class_list:
            self.event_weights[cl] = []
            weights = (self.exp_events_per_energy_bin[cl] / self.selected_events[cl]).si
            self.event_weights[cl] = weights[np.digitize(np.log10(self.mc_energies[cl]),
                                                    self.energy_bin_edges[cl])-1]

            self.event_weights[cl] = np.array(self.event_weights[cl])

        return self.event_weights

    def get_sensitivity(self, min_n=10, max_background_ratio=.05,
                        r_on=.3*u.deg, r_off=5*u.deg,
                        signal_list=("g"), verbose=False):
        """
        finally calculates the sensitivity to a point-source

        Parameters
        ----------
        min_n : integer, optional (default: 10)
            minimum number of events per energy bin -- if the number is smaller, scale up
            all events to sum up to this
        max_background_ratio : float, optional (default: 0.05)
            maximal background contamination per bin -- if fraction of protons in a bin
            is larger than this, scale up the gammas events accordingly
        r_on, r_off : floats, optional (defaults: 0.3, 5)
            radii of the on and off region considered for the significance calculation
        verbose : bool, optional
            print some statistics for every energy bin

        Returns
        -------
        sensitivities : astropy.table.Table
            the sensitivity for every energy bin of `.energy_bin_edges[signal_list[0]]`
        """

        # the area-ratio of the on- and off-region
        # A_on = r_on**2 * pi
        # A_off = r_off**2 * pi - r_on**2 * pi
        #       = (r_off**2 - r_on**2) * pi
        # alpha = A_on / A_off
        alpha = 1/(((r_off/r_on)**2)-1)

        # sensitivities go in here
        sensitivities = Table(names=("Energy MC", "Sensitivity"))
        sensitivities["Energy MC"].unit = self.energy_unit
        sensitivities["Sensitivity"].unit = self.flux_unit

        # loop over all energy bins
        for elow, ehigh in zip(10**(self.energy_bin_edges[signal_list[0]][:-1]),
                               10**(self.energy_bin_edges[signal_list[0]][1:])):

            # N_events[backgr/signal][on/off region]
            N_events = np.array([[0., 0.], [0., 0.]])
            N_backgr, N_signal = N_events

            # count the (weights of the) events in the on and off regions for this
            # energy bin
            for cl in self.class_list:
                # all events that have a smaller angular offset than `r_on`
                on_mask = (self.off_angles[cl] < r_on)
                # all events that have a smaller angular offset than `r_off` but are
                # not in the on-region
                off_mask = (self.off_angles[cl] < r_off) ^ on_mask
                # single out the events in this energy bin
                e_mask = (self.mc_energies[cl] > elow) & (self.mc_energies[cl] < ehigh)
                # is this channel signal or background
                is_sig = int(cl in signal_list)
                # count the events as the sum of their weights with all masks applied
                N_events[is_sig][0] += np.sum(self.event_weights[cl][e_mask & on_mask])
                N_events[is_sig][1] += np.sum(self.event_weights[cl][e_mask & off_mask])

            # if we have no gammas in the on region, there is no sensitivity
            if N_signal[0] == 0:
                continue

            # find the scaling factor for the gamma events that gives a 5 sigma discovery
            # in this energy bin
            scale = minimize(diff_to_X_sigma, [1e-3],
                             args=(N_signal, N_backgr, alpha),
                             method='L-BFGS-B', bounds=[(1e-4, None)],
                             options={'disp': False}
                             ).x[0]

            # scale up the gamma events by this factor
            N_signal = N_signal * scale

            if verbose:
                print("e low {}\te high {}".format(np.log10(elow),
                                                   np.log10(ehigh)))

            # check if there are sufficient events in this energy bin
            scale_a = check_min_N(N_signal, N_backgr, min_n, verbose)

            # and scale the gamma events accordingly if not
            if scale_a > 1:
                N_signal = N_signal * scale_a
                scale *= scale_a

            # check if the relative amount of protons in this bin is sufficiently small
            scale_r = check_background_contamination(N_signal, N_backgr,
                                                     max_background_ratio, verbose)
            # and scale the gamma events accordingly if not
            if scale_r > 1:
                N_signal = N_signal * scale_r
                scale *= scale_r

            # get the flux at the bin centre
            flux = Eminus2((elow+ehigh)/2.)
            # and scale it up by the determined factor
            sensitivity = flux*scale

            # store results in arrays
            sensitivities.add_row([(elow+ehigh)/2., sensitivity.to(self.flux_unit)])

            if verbose:
                print("sensitivity: ", sensitivity)
                print("Non:", N_signal[0]+N_backgr[1])
                print("Noff:", N_signal[0]+N_backgr[1])
                print("  {}, {}".format(N_signal, N_backgr))
                print("alpha:", alpha)
                print("sigma:", sigma_lima(N_signal[0]+N_backgr[1],
                                           N_signal[0]+N_backgr[1], alpha=alpha))

                print()

        return sensitivities

    def calculate_sensitivities(self,
                                # arguments for `get_effective_areas`
                                n_simulated_events=None,
                                spectra=None,
                                generator_energy_hists={},
                                generator_areas={'g': np.tau/2*(1000*u.m)**2,
                                          'p': np.tau/2*(2000*u.m)**2},

                                # arguments for `get_expected_events`
                                rates={'g': Eminus2, 'p': CR_background_rate},
                                extensions={'p': 6*u.deg},
                                observation_time=50*u.h,

                                # arguments for `get_sensitivity`
                                min_n=10, max_prot_ratio=.05,
                                r_on=.3*u.deg, r_off=5*u.deg
                                ):
        """
        wrapper that calls all functions to calculate the point-source sensitivity

        Parameters
        ----------
        cf. corresponding functions

        Returns
        -------
        sensitivities : astropy.table.Table
            the sensitivity for every energy bin of `.bin_edges_gam`

        """
        print("calling 'get_effective_areas'")
        self.get_effective_areas(n_simulated_events,
                                 spectra,
                                 generator_energy_hists,
                                 generator_areas)
        print("calling 'get_expected_events'")
        self.get_expected_events(rates,
                                 extensions,
                                 observation_time)
        print("calling 'scale_events_to_expected_events'")
        self.scale_events_to_expected_events()
        print("calling 'get_sensitivity'")
        return self.get_sensitivity(min_n, max_prot_ratio, r_on, r_off)


def check_min_N(N_signal, N_backgr, min_N, verbose=True):
    """
    check if there are sufficenly many events in this energy bin and calculates scaling
    parameter if not

    Parameters
    ----------
    N_signal, N_backgr : shape (2) numpy arrays
        the on (index 0) and off (index 1) counts for gamma and proton events
    min_n : integer
        minimum number of desired events; if too low, scale up to this number
    verbose : bool, optional (default: True)
        print some numbers if true

    Returns
    -------
    scale_a : float
        factor to scale up gamma events if insuficently many events present
    """

    if np.sum(N_signal) + np.sum(N_backgr) < min_N:
        scale_a = (min_N-np.sum(N_backgr)) / np.sum(N_signal)

        if verbose:
            print("  N_tot too small: {}, {}".format(N_signal, N_backgr))
            print("  scale_a:", scale_a)

        return scale_a
    else:
        return 1


def check_background_contamination(N_signal, N_backgr, max_prot_ratio, verbose=True):
    """
    check if there are sufficenly few proton events in this energy bin and calculates
    scaling parameter if not

    Parameters
    ----------
    N_signal, N_backgr : shape (2) numpy arrays
        the on (index 0) and off (index 1) counts for gamma and proton events
    max_prot_ratio : float
        maximum allowed proton contamination
    verbose : bool, optional (default: True)
        print some numbers if true

    Returns
    -------
    scale_r : float
        factor to scale up gamma events if too high proton contamination
    """

    N_backgr_tot = np.sum(N_backgr)
    N_signal_tot = np.sum(N_signal)
    N_tot = N_signal_tot + N_backgr_tot
    if N_backgr_tot / N_tot > max_prot_ratio:
        scale_r = (1/max_prot_ratio - 1) * N_backgr_tot / N_signal_tot
        if verbose:
            print("  too high proton contamination: {}, {}".format(N_signal, N_backgr))
            print("  scale_r:", scale_r)
        return scale_r
    else:
        return 1
