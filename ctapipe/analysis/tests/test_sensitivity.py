import numpy as np
import astropy.units as u
import scipy.stats

from ctapipe.analysis.sensitivity import SensitivityPointSource
from ctapipe.analysis.sensitivity import (check_min_N, check_background_contamination,
                                          CR_background_rate, Eminus2, sigma_lima,
                                          make_mock_event_rate, crab_source_rate)


def test_check_min_N(n_1=2, n_2=2):
    N = [n_1, n_2]

    min_N = 10

    scale = check_min_N(N, alpha=1, min_N=min_N)

    assert sum(N) >= min_N


def test_check_background_contamination(n_1=2, n_2=2):
    N = [n_1, n_2]

    max_prot_ratio = .1

    scale = check_background_contamination(N, alpha=1, max_prot_ratio=max_prot_ratio)

    assert N[1] / sum(N) <= max_prot_ratio


def test_SensitivityPointSource_effective_area():

    # areas used in the event generator
    gen_area_g = np.pi*(1000*u.m)**2
    gen_area_p = np.pi*(2000*u.m)**2

    # energy list of "selected" events
    energy_sel_gammas = np.logspace(2, 6, 200, False) * u.TeV
    energy_sel_electr = np.logspace(2, 6, 200, False) * u.TeV
    energy_sel_proton = np.logspace(2, 6, 400, False) * u.TeV

    # energy list of "generated" events
    energy_sim_gammas = np.logspace(2, 6, 400, False) * u.TeV
    energy_sim_electr = np.logspace(2, 6, 400, False) * u.TeV
    energy_sim_proton = np.logspace(2, 6, 800, False) * u.TeV

    # binning for the energy histograms
    energy_edges = np.logspace(2, 6, 41) * u.TeV

    # energy histogram for the generated events
    energy_sim_hist_gamma = np.histogram(energy_sim_gammas,
                                         bins=energy_edges)[0]
    energy_sim_hist_elect = np.histogram(energy_sim_electr,
                                         bins=energy_edges)[0]
    energy_sim_hist_proton = np.histogram(energy_sim_proton,
                                          bins=energy_edges)[0]

    # constructer gets fed with energy lists and desired energy binning
    sens = SensitivityPointSource(
                    mc_energies={'g': energy_sel_gammas,
                                 'p': energy_sel_proton,
                                 'e': energy_sel_electr},
                    energy_bin_edges={'g': energy_edges,
                                      'p': energy_edges,
                                      'e': energy_edges})

    # effective areas
    sens.get_effective_areas(
                    generator_energy_hists={'g': energy_sim_hist_gamma,
                                            'p': energy_sim_hist_proton,
                                            'e': energy_sim_hist_elect},
                    generator_areas={'g': gen_area_g,
                                     'p': gen_area_p,
                                     'e': gen_area_g})

    # midway result are the effective areas
    eff_a = sens.effective_areas
    eff_area_g, eff_area_p, eff_area_e = eff_a['g'], eff_a['p'], eff_a['e']

    # "selected" events are half of the "generated" events
    # so effective areas should be half of generator areas, too
    np.testing.assert_allclose(eff_area_g.value, gen_area_g.value/2)
    np.testing.assert_allclose(eff_area_p.value, gen_area_p.value/2)


def test_SensitivityPointSource_sensitivity_MC():

    alpha = 1.
    n_sig = 20
    n_bgr = 10
    energy_bin_edges = np.array([.1, 10])*u.TeV  # one bin with the bin-centre at 1 TeV
    energies_sig = np.ones(n_sig)
    energies_bgr = np.ones(n_bgr)

    sens = SensitivityPointSource(
            {'s': energies_sig*u.TeV,   # all events have 1 TeV energy
             'b': energies_bgr*u.TeV})  # all events have 1 TeV energy
    sens.event_weights = {'s': energies_sig, 'b': energies_bgr}
    sensitivity = sens.get_sensitivity(alpha=alpha, signal_list=("s"), mode="MC",
                                       sensitivity_source_flux=crab_source_rate,
                                       sensitivity_energy_bin_edges=energy_bin_edges)

    # the ratio between sensitivity and reference flux (i.e. from the crab nebula) should
    # be the scaling factor that needs to be applied to n_sig to produce a 5 sigma result
    # in the lima formula
    ratio = sensitivity["Sensitivity"][0]/(crab_source_rate(1*u.TeV)).value
    np.testing.assert_allclose([sigma_lima(ratio*n_sig+alpha*n_bgr, n_bgr, alpha)], [5])


def test_SensitivityPointSource_sensitivity_data():

    alpha = 1.
    n_on = 30
    n_off = 10
    energy_bin_edges = np.array([.1, 10])*u.TeV  # one bin with the bin-centre at 1 TeV
    energies_sig = np.ones(n_on)
    energies_bgr = np.ones(n_off)

    sens = SensitivityPointSource(
            {'s': energies_sig*u.TeV,   # all events have 1 TeV energy
             'b': energies_bgr*u.TeV})  # all events have 1 TeV energy

    sensitivity = sens.get_sensitivity(alpha=alpha, signal_list=("s"), mode="data",
                                       sensitivity_source_flux=crab_source_rate,
                                       sensitivity_energy_bin_edges=energy_bin_edges)

    # the ratio between sensitivity and reference flux (i.e. from the crab nebula) should
    # be the scaling factor that needs to be applied to n_sig=n_on-n_off*alpha to produce
    # a 5 sigma result  in the lima formula
    ratio = sensitivity["Sensitivity"][0]/(crab_source_rate(1*u.TeV)).value
    np.testing.assert_allclose([sigma_lima(ratio*(n_on-n_off*alpha)+n_off*alpha,
                                           n_off, alpha)], [5])


def test_generate_toy_timestamps():
    from scipy import signal
    gamma_light_curve = signal.gaussian(5000, 700) * 1000

    tmin, tmax = 0, 5

    n_throws = 500

    time_stamps = SensitivityPointSource.\
        generate_toy_timestamps(light_curves={'g': gamma_light_curve,
                                              'p': n_throws},
                                time_window=(tmin, tmax))

    assert len(time_stamps['g']) == int(np.sum(gamma_light_curve))
    assert len(time_stamps['p']) == n_throws
    for st in time_stamps.values():
        assert (np.min(st) >= tmin) and (np.max(st) <= tmax)
        np.testing.assert_allclose(np.mean(st), 2.5, atol=.25)


def test_draw_events_with_flux_weight():

    np.random.seed(1)

    e_min = 20
    e_max = 50
    n_draws = 10000

    energy = np.random.uniform(e_min, e_max, 50000)

    gamma_old = 0  # since events are drawn from np.random.uniform
    gamma_new = 3

    # this is a simple approach on how to determine event weights (i.e. probabilities to
    # get picked in the  random draw) to generate a set of events with a different energy
    # spectrum
    #
    # Parameters
    # ----------
    # energy : numpy array
    #     the energies of the MC events
    # gamma_new : float
    #     the spectral index the drawn set of events is supposed to follow
    # gamma_old : float
    #     the spectral index the MC events have been generated with
    #     this is zero here but usually 2 in MC generators
    #
    # Returns
    # -------
    # weights : numpy array
    #     list of event weights normalised to 1
    #     to be used as PDF in `np.random.choice`
    energy_weights = energy**(gamma_old - gamma_new)
    energy_weights /= np.sum(energy_weights)

    indices = SensitivityPointSource.draw_events_from_flux_weight(
                        {'g': energy},
                        {'g': energy_weights},
                        {'g': n_draws})

    assert len(energy[indices['g']]) == n_draws
    # TODO come up with more tests...


def test_draw_events_from_flux_histogram():

    np.random.seed(1)

    e_min = 20
    e_max = 50
    n_bins = 30
    n_draws = 5000

    energy_edges = np.linspace(e_min, e_max, n_bins+1, True) * u.TeV
    energy = np.random.uniform(e_min, e_max, 50000) * u.TeV

    target_distribution = make_mock_event_rate(lambda e: (e/u.TeV)**-3,
                                               energy_edges,
                                               norm=n_draws, log_e=False)

    indices = SensitivityPointSource.draw_events_from_flux_histogram(
                        {'g': energy},
                        {'g': target_distribution},
                        {'g': energy_edges})

    hist, _ = np.histogram(energy[indices['g']], bins=energy_edges[::])

    # checking the χ² between `target_distribution` and the drawn one
    chisquare = scipy.stats.chisquare(target_distribution, hist)[0]
    # the test that the reduced χ² is close to 1 (tollorance of 1)
    np.testing.assert_allclose([chisquare/n_bins], [1], atol=0.1)
