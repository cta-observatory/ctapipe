import numpy as np
from numpy import allclose
import astropy.units as u

from ctapipe.analysis.Sensitivity import *
from ctapipe.analysis.Sensitivity import check_min_N, check_background_contamination, \
                                         CR_background_rate, Eminus2


def test_check_min_N():
    N_g = np.array([2., 1.])
    N_p = np.array([0., 1.])

    min_N = 10

    scale = check_min_N(N_g, N_p, min_N)
    N_g = N_g * scale

    assert sum(N_g + N_p) == min_N


def test_check_background_contamination():
    N_g = np.array([2., 1.])
    N_p = np.array([0., 1.])

    max_prot_ratio = .1

    scale = check_background_contamination(N_g, N_p, max_prot_ratio)
    N_g = N_g * scale

    assert sum(N_p) / sum(N_g + N_p) == max_prot_ratio


def test_Sensitivity_PointSource():

    # areas used in the event generator
    gen_area_g = np.tau/2*(1000*u.m)**2
    gen_area_p = np.tau/2*(2000*u.m)**2

    # energy list of "selected" events
    energy_sel_gamma  = np.logspace(2, 6, 200, False)
    energy_sel_elect  = np.logspace(2, 6, 200, False)
    energy_sel_proton = np.logspace(2, 6, 400, False)

    # energy list of "generated" events
    energy_sim_gamma  = np.logspace(2, 6, 400, False)
    energy_sim_elect  = np.logspace(2, 6, 400, False)
    energy_sim_proton = np.logspace(2, 6, 800, False)

    # angular distance of the events from the "point-source"
    # (randomise the order so they don't align with the energy arrays)
    angles_gamma  = np.random.choice(np.logspace(-3, 1, 200)    , 200, False) * u.deg
    angles_elect  = np.random.choice(np.logspace(-3, 1, 200)    , 200, False) * u.deg
    angles_proton = np.random.choice(np.linspace(1e-3, 1e1, 400), 400, False) * u.deg

    # binning for the energy histograms
    energy_edges = np.linspace(2, 6, 41)

    # energy histogram for the generated events
    energy_sim_hist_gamma = np.histogram(np.log10(energy_sim_gamma),
                                         bins=energy_edges)[0]
    energy_sim_hist_elect = np.histogram(np.log10(energy_sim_elect),
                                         bins=energy_edges)[0]
    energy_sim_hist_proton = np.histogram(np.log10(energy_sim_proton),
                                          bins=energy_edges)[0]

    # constructer gets fed with energy and angular offset lists and desired energy binning
    Sens = Sensitivity_PointSource(
                    mc_energies={'g': energy_sel_gamma, 'p':energy_sel_proton,
                                 'e': energy_sel_elect},
                    off_angles={"g": angles_gamma, "p": angles_proton, 'e': angles_elect},
                    energy_bin_edges={"g": energy_edges,
                                      "p": energy_edges,
                                      'e': energy_edges},
                    energy_unit=u.GeV, flux_unit=u.erg/(u.m**2*u.s))

    # wrapper for various internal functions
    # spits out energy | sensitivity table
    sensitivities = Sens.calculate_sensitivities(
                    generator_energy_hists={'g': energy_sim_hist_gamma,
                                            'p': energy_sim_hist_proton,
                                            'e': energy_sim_hist_elect},
                    rates={'g': Eminus2, 'p': CR_background_rate, 'e': Eminus2},
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


def test_generate_toy_timestamps():
    from scipy import signal
    gamma_light_curve = signal.gaussian(5000, 700) * 1000

    tmin, tmax = 0, 5

    NP = 500

    time_stamps = Sensitivity_PointSource.\
        generate_toy_timestamps(light_curve={'g': gamma_light_curve,
                                             'p': NP},
                                time_window=(tmin, tmax))

    assert len(time_stamps['g']) == int(np.sum(gamma_light_curve))
    assert len(time_stamps['p']) == NP
    for st in time_stamps.values():
        assert (np.min(st) >= tmin) and (np.max(st) <= tmax)
        np.testing.assert_allclose(np.mean(st), 2.5, atol=.1)


def Emin2toEmin3(e):
    e_w = e**-1
    return e_w / np.sum(e_w)

def Emin2toFlat(e):
    e_w = e**2
    return e_w / np.sum(e_w)

def FlattoFlat(e):
    e_w = e**0
    return e_w / np.sum(e_w)


def test_draw_events_with_flux_weight():

    Emin = 0
    Emax = 50
    Nbin = 50

    energy_edges = np.linspace(Emin, Emax, Nbin, True)

    energy_sel_gamma = np.random.uniform(Emin, Emax, 50000)

    sens = Sensitivity_PointSource(
                    mc_energies={'g': energy_sel_gamma},
                    off_angles={},
                    energy_bin_edges={"g": energy_edges},
                    energy_unit=u.GeV, flux_unit=u.erg/(u.m**2*u.s))

    indices = sens.draw_events_with_flux_weight({'g': FlattoFlat}, {'g': 1000})

    from matplotlib import pyplot as plt
    plt.figure()
    plt.hist(energy_sel_gamma[indices['g']], bins=energy_edges[::])
    plt.show()

test_draw_events_with_flux_weight()
