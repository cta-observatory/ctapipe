import numpy as np
import astropy.units as u

from ctapipe.analysis.Sensitivity import *
from ctapipe.analysis.Sensitivity import check_min_N, check_background_contamination


def test_check_min_N():
    N_g = [2, 1]
    N_p = [0, 1]

    min_N = 10

    scale = check_min_N(N_g, N_p, min_N)
    N_g = N_g * scale

    assert sum(N_g + N_p) == min_N


def test_check_background_contamination():
    N_g = [2, 1]
    N_p = [0, 1]

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
    energy_sel_proton = np.logspace(2, 6, 200, False)

    # energy list of "generated" events
    energy_sim_gamma  = np.logspace(2, 6, 400, False)
    energy_sim_proton = np.logspace(2, 6, 400, False)

    # angular distance of the events from the "point-source"
    angles_gamma  = np.logspace(-3, 1, 200)
    angles_proton = np.linspace(1e-3, 1e1, 400)

    # binning for the energy histograms
    energy_edges = np.linspace(2, 6, 41)

    # energy histogram for the generated events
    energy_sim_hist_gamma = np.histogram(np.log10(energy_sim_gamma),
                                         bins=energy_edges)[0]
    energy_sim_hist_proton = np.histogram(np.log10(energy_sim_proton),
                                          bins=energy_edges)[0]

    # constructer gets fed with energy and angular offset lists and desired energy binning
    Sens = Sensitivity_PointSource(energy_sel_gamma, energy_sel_proton,
                                   angles_gamma, angles_proton,
                                   energy_edges, energy_edges,
                                   energy_unit=u.GeV, flux_unit=u.erg/(u.m**2*u.s))

    # wrapper for various internal functions
    # spits out energy | sensitivity table
    sensitivities = Sens.calculate_sensitivities(gen_energy_gamma=energy_sim_hist_gamma,
                                                 gen_energy_proton=energy_sim_hist_proton,
                                                 gen_area_gamma=gen_area_g,
                                                 gen_area_proton=gen_area_p)

    # midway result are the effective areas
    eff_area_g, eff_area_p = Sens.eff_area_gam, Sens.eff_area_pro

    # "selected" events are half of the "generated" events
    # so effective areas should be half of generator areas, too
    assert (eff_area_g == gen_area_g/2).all()
    assert (eff_area_p == gen_area_p/2).all()
