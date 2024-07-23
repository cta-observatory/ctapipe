import astropy.units as u
import numpy as np
import pytest
from astropy.table import QTable, vstack

from ctapipe.irf import BackgroundRate2dMaker, EventPreProcessor


@pytest.fixture(scope="session")
def irf_events_table():
    N1 = 1000
    N2 = 100
    N = N1 + N2
    epp = EventPreProcessor()
    tab = epp.make_empty_table()
    units = {c: tab[c].unit for c in tab.columns}

    empty = np.zeros((len(tab.columns), N)) * np.nan
    e_tab = QTable(data=empty.T, names=tab.colnames, units=units)
    # Setting values following pyirf test in pyirf/irf/tests/test_background.py
    e_tab["reco_energy"] = np.append(np.full(N1, 1), np.full(N2, 2)) * u.TeV
    e_tab["reco_source_fov_offset"] = (np.zeros(N) * u.deg,)

    ev = vstack([e_tab, tab], join_type="exact", metadata_conflicts="silent")
    return ev


def test_make_2d_bkg(irf_events_table):
    bkgMkr = BackgroundRate2dMaker()

    bkg_hdu = bkgMkr.make_bkg_hdu(events=irf_events_table, obs_time=1 * u.s)
    assert bkg_hdu.data["BKG"].shape == (1, 1, 20)
