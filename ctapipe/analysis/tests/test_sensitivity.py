import numpy as np
import astropy.units as u
import scipy.stats

from ctapipe.analysis.sensitivity import SensitivityPointSource
from ctapipe.analysis.sensitivity import (check_min_N, check_background_contamination,
                                          CR_background_rate, Eminus2, sigma_lima,
                                          make_mock_event_rate, crab_source_rate)


def test_check_min_N():
    N = [2, 2]

    min_N = 10

    scale = check_min_N(N, min_N=min_N)

    assert sum(N) == min_N


def test_check_background_contamination():
    N = [2, 2]

    max_prot_ratio = .1

    scale = check_background_contamination(N, max_prot_ratio=max_prot_ratio)

    assert N[1] / sum(N) == max_prot_ratio


def test_SensitivityPointSource_effective_area():

    # areas used in the event generator
    gen_area_g = np.pi*(1000*u.m)**2
    gen_area_p = np.pi*(2000*u.m)**2

    # energy list of "selected" events
    energy_sel_gamma  = np.logspace(2, 6, 200, False) * u.TeV
    energy_sel_elect  = np.logspace(2, 6, 200, False) * u.TeV
    energy_sel_proton = np.logspace(2, 6, 400, False) * u.TeV

    # energy list of "generated" events
    energy_sim_gamma  = np.logspace(2, 6, 400, False) * u.TeV
    energy_sim_elect  = np.logspace(2, 6, 400, False) * u.TeV
    energy_sim_proton = np.logspace(2, 6, 800, False) * u.TeV

    # binning for the energy histograms
    energy_edges = np.logspace(2, 6, 41) * u.TeV

    # energy histogram for the generated events
    energy_sim_hist_gamma = np.histogram(energy_sim_gamma,
                                         bins=energy_edges)[0]
    energy_sim_hist_elect = np.histogram(energy_sim_elect,
                                         bins=energy_edges)[0]
    energy_sim_hist_proton = np.histogram(energy_sim_proton,
                                          bins=energy_edges)[0]

    # constructer gets fed with energy and angular offset lists and desired energy binning
    Sens = SensitivityPointSource(
                    mc_energies={'g': energy_sel_gamma,
                                 'p': energy_sel_proton,
                                 'e': energy_sel_elect},
                    energy_bin_edges={"g": energy_edges,
                                      "p": energy_edges,
                                      'e': energy_edges})

    # effective areas
    Sens.get_effective_areas(
                    generator_energy_hists={'g': energy_sim_hist_gamma,
                                            'p': energy_sim_hist_proton,
                                            'e': energy_sim_hist_elect},
                    generator_areas={'g': gen_area_g,
                                     'p': gen_area_p,
                                     'e': gen_area_g})

    # midway result are the effective areas
    eff_a = Sens.effective_areas
    eff_area_g, eff_area_p, eff_area_e = eff_a['g'], eff_a['p'], eff_a['e']

    # "selected" events are half of the "generated" events
    # so effective areas should be half of generator areas, too
    np.testing.assert_allclose(eff_area_g.value, gen_area_g.value/2)
    np.testing.assert_allclose(eff_area_p.value, gen_area_p.value/2)


def test_SensitivityPointSource_sensitivity_MC():

    alpha = 1.
    Nsig = 20
    Nbgr = 10
    energy_bin_edges = np.array([.1, 10])*u.TeV  # one bin with the bin-centre at 1 TeV
    energies_sig = np.ones(Nsig)
    energies_bgr = np.ones(Nbgr)

    sens = SensitivityPointSource(
            {'s': energies_sig*u.TeV,   # all events have 1 TeV energy
             'b': energies_bgr*u.TeV})  # all events have 1 TeV energy
    sens.event_weights = {'s': energies_sig, 'b': energies_bgr}
    sensitivity = sens.get_sensitivity(alpha=alpha, signal_list=("s"), mode="MC",
                                       sensitivity_source_flux=crab_source_rate,
                                       sensitivity_energy_bin_edges=energy_bin_edges)

    # the ratio between sensitivity and reference flux (i.e. from the crab nebula) should
    # be the scaling factor that needs to be applied on Nsig to produce a 5 sigma result
    # in the lima formula
    ratio = sensitivity["Sensitivity"][0]/(crab_source_rate(1*u.TeV)).value
    np.testing.assert_allclose([sigma_lima(ratio*Nsig+alpha*Nbgr, Nbgr, alpha)], [5])


def test_SensitivityPointSource_sensitivity_data():

    alpha = 1.
    Non = 30
    Noff = 10
    energy_bin_edges = np.array([.1, 10])*u.TeV  # one bin with the bin-centre at 1 TeV
    energies_sig = np.ones(Non)
    energies_bgr = np.ones(Noff)

    sens = SensitivityPointSource(
            {'s': energies_sig*u.TeV,   # all events have 1 TeV energy
             'b': energies_bgr*u.TeV})  # all events have 1 TeV energy

    sensitivity = sens.get_sensitivity(alpha=alpha, signal_list=("s"), mode="data",
                                       sensitivity_source_flux=crab_source_rate,
                                       sensitivity_energy_bin_edges=energy_bin_edges)

    # the ratio between sensitivity and reference flux (i.e. from the crab nebula) should
    # be the scaling factor that needs to be applied on Nsig to produce a 5 sigma result
    # in the lima formula
    ratio = sensitivity["Sensitivity"][0]/(crab_source_rate(1*u.TeV)).value
    np.testing.assert_allclose([sigma_lima(ratio*(Non-Noff*alpha)+Noff,
                                           Noff, alpha)], [5])


def test_generate_toy_timestamps():
    from scipy import signal
    gamma_light_curve = signal.gaussian(5000, 700) * 1000

    tmin, tmax = 0, 5

    Nthrows = 500

    time_stamps = SensitivityPointSource.\
        generate_toy_timestamps(light_curves={'g': gamma_light_curve,
                                              'p': Nthrows},
                                time_window=(tmin, tmax))

    assert len(time_stamps['g']) == int(np.sum(gamma_light_curve))
    assert len(time_stamps['p']) == Nthrows
    for st in time_stamps.values():
        assert (np.min(st) >= tmin) and (np.max(st) <= tmax)
        np.testing.assert_allclose(np.mean(st), 2.5, atol=.1)


def test_draw_events_with_flux_weight():

    Emin = 20
    Emax = 50
    Nbin = 50
    Ndraws = 10000

    def weight_from_energy(e, gamma_new=3, gamma_old=2):
        """
        use this to determine event weights (i.e. probabilities to get picked in the
        random draw

        Parameters
        ----------
        e : numpy array
            the energies of the MC events
        gamma_new : float
            the spectral index the drawn set of events are supposed to follow
        gamma_old : float
            the spectral index the MC events have been generated with

        Returns
        -------
        weights : numpy array
            list of event weights normalised to 1
            to be used as PDF in `np.random.choice`
        """
        e_w = e**(gamma_old - gamma_new)
        return e_w / np.sum(e_w)

    energy_sel_gamma = np.random.uniform(Emin, Emax, 50000)

    indices = SensitivityPointSource.draw_events_from_flux_weight(
                        {'g': energy_sel_gamma},
                        {'g': weight_from_energy(energy_sel_gamma, 2, 0)},
                        {'g': Ndraws})

    assert len(energy_sel_gamma[indices['g']]) == Ndraws
    # TODO come up with more tests...


def test_draw_events_from_flux_histogram():
    Emin = 20
    Emax = 50
    Nbin = 30
    Ndraws = 5000

    energy_edges = np.linspace(Emin, Emax, Nbin+1, True) * u.TeV
    energy_sel_gamma = np.random.uniform(Emin, Emax, 50000) * u.TeV

    target_distribution = make_mock_event_rate(lambda e: (e/u.TeV)**-3,
                                               energy_edges,
                                               norm=Ndraws, log_e=False)

    indices = SensitivityPointSource.draw_events_from_flux_histogram(
                        {'g': energy_sel_gamma},
                        {'g': target_distribution},
                        {'g': energy_edges})

    hist, _ = np.histogram(energy_sel_gamma[indices['g']], bins=energy_edges[::])

    # checking the χ² between `target_distribution` and the drawn one
    chisquare = scipy.stats.chisquare(target_distribution, hist)[0]
    # the test that the reduced χ² is close to 1 (tollorance of 1)
    np.testing.assert_allclose([chisquare/Nbin], [1], atol=1)
