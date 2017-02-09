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
        if set, uses this as the unit of the numpy array
        ought to be of same dimension of the quantities in the list (there is no test)

    Returns
    -------
    a : quantified numpy array
    """

    if unit is None:
        unit = arr[0].unit
        return (np.array([a.to(unit).value for a in arr])*unit).si
    else:
        return np.array([a.to(unit).value for a in arr])*unit


def crab_source_rate(energy):
    ''' function for a pseudo-Crab point source rate
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
    boring old unnormalised E^-2 spectrum

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


def diff_to_X_sigma(scale, N_g, N_p, alpha, X=5):
    """
    calculates the significance according to `sigma_lima` and returns the squared
    difference to X. To be used in a minimiser that determines the necessary source
    intensity for a detection of given significance of `X` sigma

    Parameters
    ----------
    scale : python list with a single float
        this is the variable in the minimisation procedure
        it scales the number of gamma events
    N_g, N_p : shape (2) numpy arrays
        the on (index 0) and off (index 1) counts for gamma and proton events
        the gamma events are to be scaled by `scale[0]`
    alpha : float
        the ratio of the on and off areas
    X : float, optional (deflaut: 5)
        target significance in multiples of "sigma"

    Returns
    -------
    (sigma-X)**2 : float
        squared difference of the significance to `X`
        minimise this function for the number of gamma events needed for your desired
        significance of `X` sigma
    """

    Non = N_p[0] + N_g[0] * scale[0]
    Noff = N_p[1] + N_g[1] * scale[0]
    sigma = sigma_lima(Non, Noff, alpha)
    return (sigma-X)**2


class Sensitivity_PointSource():
    """
    class to calculate the sensitivity to a known point-source
    TODO:
        • add extended source?
        • add pseude experiment for low exposure times?
        • make "background" more flexible / particle agnostic so you can add electrons
          and/or muons?
    """
    def __init__(self, mc_energy_gamma, mc_energy_proton,
                 off_angles_g, off_angles_p,
                 bin_edges_gamma, bin_edges_proton,
                 energy_unit=u.GeV, flux_unit=u.GeV / (u.m**2*u.s)):
        """
        constructor, simply sets some initial parameters

        Parameters
        ----------
        mc_energy_gamma, mc_energy_proton: quantified numpy arrays
            list of simulated energies of the selected gammas and protons
        off_angles_g, off_angles_p : numpy arrays
            list of offset angles between the reconstructed direction and the point-source
            direction for gamma and proton events
        bin_edges_gamma, bin_edges_proton : numpy arrays
            list of the bin edges for the various histograms for gammas and protons
            assumes binning in log10(energy)
        energy_unit : astropy quantity, optional (default: u.GeV)
            your favourite energy unit
        flux_unit : astropy quantity, optional (default: u.GeV / (u.m**2 * u.s))
            your favourite differential flux unit

        """

        self.mc_energy_gam = mc_energy_gamma
        self.mc_energy_pro = mc_energy_proton
        self.off_angles_g = off_angles_g
        self.off_angles_p = off_angles_p
        self.bin_edges_gam = bin_edges_gamma
        self.bin_edges_pro = bin_edges_proton

        self.energy_unit = energy_unit
        self.flux_unit = flux_unit

    def get_effective_areas(self, n_simulated_gamma=None, n_simulated_proton=None,
                            spectrum_gamma=Eminus2, spectrum_proton=Eminus2,
                            gen_energy_gamma=None, gen_energy_proton=None,
                            gen_area_gamma=np.tau/2*(1000*u.m)**2,
                            gen_area_proton=np.tau/2*(2000*u.m)**2):
        """
        calculates the effective areas for gammas and protons and stores them in the
        class instance

        Parameters
        ----------
        n_simulated_gamma, n_simulated_proton : integers, optional (defaults: None)
            number of gamma and proton events used in the MC simulation
            either use this with the `spectrum_gamma` and `spectrum_proton` parameters
            or directly the `gen_energy_gamma` and `gen_energy_proton` parameters
        spectrum_gamma, spectrum_proton : functions, optional (default: Eminus2)
            function object for the differential generator flux of the gamma and proton
            events
        gen_energy_gamma, gen_energy_proton : numpy arrays, optional (defaults: None)
            energy histogram of the generated gammas and protons binned according to
            `.bin_edges_gam` and `.bin_edges_pro`
            either use these directly or generate them with the `n_simulated_...` and
            `spectrum_...` parameters
        gen_area_gamma, gen_area_proton : astropy quantities, optional (defaults:
        pi*(1 km)**2 and pi*(2 km)**2)
            the area within which the shower impact position was generated

        Returns
        -------
        eff_area_gam, eff_area_pro : numpy arrays
            histograms of the effective areas of gammas and protons binned according to
            `.bin_edges_gam` and `.bin_edges_pro`
        """

        if gen_energy_gamma is None:
            gen_energy_gamma = make_mock_event_rate(
                            spectrum_gamma, norm=n_simulated_gamma,
                            bin_edges=self.bin_edges_gam)[0]
        if gen_energy_proton is None:
            gen_energy_proton = make_mock_event_rate(
                            spectrum_proton, norm=n_simulated_proton,
                            bin_edges=self.bin_edges_pro)[0]

        self.Sel_Gammas = np.histogram(np.log10(self.mc_energy_gam),
                                       bins=self.bin_edges_gam)[0]
        self.Sel_Proton = np.histogram(np.log10(self.mc_energy_pro),
                                       bins=self.bin_edges_pro)[0]

        Efficiency_Gammas = self.Sel_Gammas / gen_energy_gamma
        Efficiency_Proton = self.Sel_Proton / gen_energy_proton

        self.eff_area_gam = Efficiency_Gammas * gen_area_gamma
        self.eff_area_pro = Efficiency_Proton * gen_area_proton

        return self.eff_area_gam, self.eff_area_pro

    def get_expected_events(self, source_rate=Eminus2, background_rate=CR_background_rate,
                            extension_gamma=None, extension_proton=6*u.deg,
                            observation_time=50*u.h):
        """
        given a source rate and the effective area, calculates the number of expected
        events within a given observation time

        Parameters
        ----------
        source_rate, background_rate : callables, optional (default: `Eminus2` and
            `CR_background_rate`)
            functions for the differential source and background rates
        extension_gamma, extension_proton : astropy quantities, optional (defaults: None
            and 6*u.deg)
            opening angle of the view-cone the events have been generated in
            put `None` for point source
            note: if you use an extension, the flux needs to accomodate that as well
        observation_time : astropy quantity, optional (default: 50*u.h)
            length of the assumed exposure
        """

        # for book keeping
        self.observation_time = observation_time

        SourceRate = make_mock_event_rate(source_rate,
                                          bin_edges=self.bin_edges_gam)[0]
        BackgrRate = make_mock_event_rate(background_rate,
                                          bin_edges=self.bin_edges_pro)[0]

        if extension_gamma:
            omega_gam = np.tau*(1 - np.cos(extension_gamma))*u.rad**2
            SourceRate *= omega_gam
        if extension_proton:
            omega_pro = np.tau*(1 - np.cos(extension_proton))*u.rad**2
            BackgrRate *= omega_pro

        try:
            self.exp_events_per_E_gam = SourceRate * observation_time * self.eff_area_gam
            self.exp_events_per_E_pro = BackgrRate * observation_time * self.eff_area_pro

            return self.exp_events_per_E_gam, self.exp_events_per_E_pro
        except AttributeError as e:
            print("did you call get_effective_areas already?")
            raise e

    def scale_events_to_expected_events(self):
        """
        produces weights to scale the selected events according to source/background
        spectrum and observation time

        Returns
        -------
        weight_g, weight_p : python lists
            weights for the selected gamma and proton events so that they are scaled to
            the number of expected events in `exp_events_per_E_gam` and
            `exp_events_per_E_pro` for every energy bin
        """

        weight_g = []
        weight_p = []
        for ev in self.mc_energy_gam:
            weight_g.append((self.exp_events_per_E_gam/self.Sel_Gammas)[
                                np.digitize(np.log10(ev), self.bin_edges_gam) - 1])
        for ev in self.mc_energy_pro:
            weight_p.append((self.exp_events_per_E_pro/self.Sel_Proton)[
                                np.digitize(np.log10(ev), self.bin_edges_pro) - 1])

        self.weight_g = np.array(weight_g)
        self.weight_p = np.array(weight_p)
        return self.weight_g, self.weight_p

    def get_sensitivity(self, min_n=10, max_prot_ratio=.05, r_on=.3, r_off=5,
                        verbose=False):
        """
        finally calculates the sensitivity to a point-source

        Parameters
        ----------
        min_n : integer, optional (default: 10)
            minimum number of events per energy bin -- if the number is smaller, scale up
            all events to sum up to this
        max_prot_ratio : float, optional (default: 0.05)
            maximal proton contamination per bin -- if fraction of protons in a bin is
            larger than this, scale up the gammas events accordingly
        r_on, r_off : floats, optional (defaults: 0.3, 5)
            radii of the on and off region considered for the significance calculation
        verbose : bool, optional
            print some statistics for every energy bin

        Returns
        -------
        sensitivities : astropy.table.Table
            the sensitivity for every energy bin of `.bin_edges_gam`
        """

        # the area-ratio of the on- and off-region
        alpha = 1/(((r_off/r_on)**2)-1)

        # sensitivities go in here
        sensitivities = Table(names=("Energy MC", "Sensitivity"))
        sensitivities["Energy MC"].unit = self.energy_unit
        sensitivities["Sensitivity"].unit = self.flux_unit

        # loob over all energy bins
        for elow, ehigh in zip(10**(self.bin_edges_gam[:-1]),
                               10**(self.bin_edges_gam[1:])):

            N_g = np.array([0, 0])
            N_p = np.array([0, 0])

            # loop over all angular distances and their weights for this energy bin
            # and count the events in the on and off regions
            # for gammas ...
            for s, w in zip(self.off_angles_g[(self.mc_energy_gam > elow) &
                                              (self.mc_energy_gam < ehigh)],
                            self.weight_g[(self.mc_energy_gam > elow) &
                                          (self.mc_energy_gam < ehigh)]):
                if s < r_off:
                    N_g[int(s > r_on)] += w

            # ... and protons
            for s, w in zip(self.off_angles_p[(self.mc_energy_pro > elow) &
                                              (self.mc_energy_pro < ehigh)],
                            self.weight_p[(self.mc_energy_pro > elow) &
                                          (self.mc_energy_pro < ehigh)]):
                if s < r_off:
                    N_p[int(s > r_on)] += w

            # if we have no gammas in the on region, there is no sensitivity
            if N_g[0] == 0:
                continue

            # find the scaling factor for the gamma events that gives a 5 sigma discovery
            # in this energy bin
            scale = minimize(diff_to_X_sigma, [1e-3],
                             args=(N_g, N_p, alpha),
                             # method='BFGS',
                             method='L-BFGS-B', bounds=[(1e-4, None)],
                             options={'disp': False}
                             ).x[0]

            # scale up the gamma events by this factor
            N_g = N_g * scale

            if verbose:
                print("e low {}\te high {}".format(np.log10(elow),
                                                   np.log10(ehigh)))

            # check if there are sufficient events in this energy bin
            scale_a = check_min_N(N_g, N_p, min_n, verbose)

            # and scale the gamma events accordingly if not
            if scale_a > 1:
                N_g = N_g * scale_a
                scale *= scale_a

            # check if the relative amount of protons in this bin is sufficiently small
            scale_r = check_background_contamination(N_g, N_p, max_prot_ratio, verbose)
            # and scale the gamma events accordingly if not
            if scale_r > 1:
                N_g = N_g * scale_r
                scale *= scale_r

            # get the flux at the bin centre
            flux = Eminus2((elow+ehigh)/2.).to(self.flux_unit)
            # and scale it up by the determined factor
            sensitivity = flux*scale

            # store results in arrays
            sensitivities.add_row([(elow+ehigh)/2., sensitivity.to(self.flux_unit)])

            if verbose:
                print("sensitivity: ", sensitivity)
                print("Non:", N_g[0]+N_p[1])
                print("Noff:", N_g[0]+N_p[1])
                print("  {}, {}".format(N_g, N_p))
                print("alpha:", alpha)
                print("sigma:", sigma_lima(N_g[0]+N_p[1], N_g[0]+N_p[1], alpha=alpha))

                print()

        return sensitivities

    def calculate_sensitivities(self,
                                # arguments for `get_effective_areas`
                                n_simulated_gamma=None, n_simulated_proton=None,
                                spectrum_gamma=Eminus2, spectrum_proton=Eminus2,
                                gen_energy_gamma=None, gen_energy_proton=None,
                                gen_area_gamma=np.tau/2*(1000*u.m)**2,
                                gen_area_proton=np.tau/2*(2000*u.m)**2,

                                # arguments for `get_expected_events`
                                source_rate=Eminus2, background_rate=CR_background_rate,
                                extension_gamma=None, extension_proton=6*u.deg,
                                observation_time=50*u.h,

                                # arguments for `get_sensitivity`
                                min_n=10, max_prot_ratio=.05, r_on=.3, r_off=5
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
        self.get_effective_areas(n_simulated_gamma, n_simulated_proton,
                                 spectrum_gamma, spectrum_proton,
                                 gen_energy_gamma, gen_energy_proton,
                                 gen_area_gamma, gen_area_proton)
        self.get_expected_events(source_rate, background_rate,
                                 extension_gamma, extension_proton,
                                 observation_time)
        self.scale_events_to_expected_events()
        return self.get_sensitivity(min_n, max_prot_ratio, r_on, r_off)


def check_min_N(N_g, N_p, min_N, verbose=True):
    """
    check if there are sufficenly many events in this energy bin and calculates scaling
    parameter if not

    Parameters
    ----------
    N_g, N_p : shape (2) numpy arrays
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

    if np.sum(N_g) + np.sum(N_p) < min_N:
        scale_a = (min_N-np.sum(N_p)) / np.sum(N_g)

        if verbose:
            print("  N_tot too small: {}, {}".format(N_g, N_p))
            print("  scale_a:", scale_a)

        return scale_a
    else:
        return 1


def check_background_contamination(N_g, N_p, max_prot_ratio, verbose=True):
    """
    check if there are sufficenly few proton events in this energy bin and calculates
    scaling parameter if not

    Parameters
    ----------
    N_g, N_p : shape (2) numpy arrays
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

    N_p_tot = np.sum(N_p)
    N_g_tot = np.sum(N_g)
    N_tot = N_g_tot + N_p_tot
    if N_p_tot / N_tot > max_prot_ratio:
        scale_r = (1/max_prot_ratio - 1) * N_p_tot / N_g_tot
        if verbose:
            print("  too high proton contamination: {}, {}".format(N_g, N_p))
            print("  scale_r:", scale_r)
        return scale_r
    else:
        return 1
