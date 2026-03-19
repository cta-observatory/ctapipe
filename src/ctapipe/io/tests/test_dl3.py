from datetime import UTC, datetime, timedelta

import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Column, Table
from astropy.time import Time
from traitlets.config import Config

from ...irf.cuts import DL2EventSelection
from ...tools.create_dl3 import DL3Tool
from ..dl3 import DL3GADFEventsWriter


@pytest.fixture
def hdu_irfs(dummy_irf_file):
    with fits.open(dummy_irf_file, checksum=True) as hdus:
        yield hdus


@pytest.fixture(scope="session")
def dl2_events_for_dl3(single_obs_gamma_diffuse_full_reco_file, dummy_cuts_file):
    tool = DL3Tool(
        config=Config(
            {
                "EventPreprocessor": {
                    "energy_reconstructor": "ExtraTreesRegressor",
                    "geometry_reconstructor": "HillasReconstructor",
                    "gammaness_reconstructor": "ExtraTreesClassifier",
                }
            }
        )
    )
    tool.dl2_file = single_obs_gamma_diffuse_full_reco_file
    tool.event_selection = DL2EventSelection(parent=tool, cuts_file=dummy_cuts_file)
    tool._configure_event_preprocessor_for_dl3()

    events = tool._load_preselected_events(chunk_size=1000)
    events = tool.event_selection.calculate_gamma_selection(
        events,
        apply_spatial_selection=False,
    )
    events = events[events["selected_gamma"]]

    meta = tool._get_observation_information()
    events["time"] = (
        Time("2020-01-01T00:00:00", scale="tai") + np.arange(len(events)) * u.s
    )
    tool.output_table_schema = [
        Column(name="reco_az", unit=u.deg),
        Column(name="reco_alt", unit=u.deg),
        Column(name="pointing_az", unit=u.deg),
        Column(name="pointing_alt", unit=u.deg),
        Column(name="time"),
        Column(name="reco_ra", unit=u.deg),
        Column(name="reco_dec", unit=u.deg),
    ]
    events = tool._make_derived_columns(events, location_subarray=meta["location"])
    return events


@pytest.fixture(scope="session")
def dl2_meta_for_dl3(single_obs_gamma_diffuse_full_reco_file, dummy_cuts_file):
    tool = DL3Tool(
        config=Config(
            {
                "EventPreprocessor": {
                    "energy_reconstructor": "ExtraTreesRegressor",
                    "geometry_reconstructor": "HillasReconstructor",
                    "gammaness_reconstructor": "ExtraTreesClassifier",
                }
            }
        )
    )
    tool.dl2_file = single_obs_gamma_diffuse_full_reco_file
    tool.event_selection = DL2EventSelection(parent=tool, cuts_file=dummy_cuts_file)
    return tool._get_observation_information()


@pytest.fixture
def dl3_writer(dl2_events_for_dl3, dl2_meta_for_dl3, hdu_irfs):
    dl3_format_optional = DL3GADFEventsWriter()

    # Load events
    dl3_format_optional.events = dl2_events_for_dl3

    # Load metadata
    dl3_format_optional.obs_id = dl2_meta_for_dl3["obs_id"]
    dl3_format_optional.pointing = dl2_meta_for_dl3["pointing"]["pointing_list"]
    dl3_format_optional.pointing_mode = dl2_meta_for_dl3["pointing"]["pointing_mode"]
    dl3_format_optional.gti = dl2_meta_for_dl3["gti"]
    dl3_format_optional.livetime_fraction = dl2_meta_for_dl3["livetime_fraction"]
    dl3_format_optional.location = dl2_meta_for_dl3["location"]
    dl3_format_optional.telescope_information = dl2_meta_for_dl3[
        "telescope_information"
    ]
    dl3_format_optional.target_information = dl2_meta_for_dl3["target"]
    dl3_format_optional.software_information = dl2_meta_for_dl3["software_version"]

    # Load IRFs
    for i in range(1, len(hdu_irfs)):
        if "HDUCLAS2" in hdu_irfs[i].header.keys():
            if hdu_irfs[i].header["HDUCLAS2"] == "EFF_AREA":
                if dl3_format_optional.aeff is None:
                    dl3_format_optional.aeff = hdu_irfs[i]
                elif "EXTNAME" in hdu_irfs[i].header and not (
                    "PROTONS" in hdu_irfs[i].header["EXTNAME"]
                    or "ELECTRONS" in hdu_irfs[i].header["EXTNAME"]
                ):
                    dl3_format_optional.aeff = hdu_irfs[i]
            elif hdu_irfs[i].header["HDUCLAS2"] == "EDISP":
                dl3_format_optional.edisp = hdu_irfs[i]
            elif hdu_irfs[i].header["HDUCLAS2"] == "PSF":
                dl3_format_optional.psf = hdu_irfs[i]
            elif hdu_irfs[i].header["HDUCLAS2"] == "BKG":
                dl3_format_optional.bkg = hdu_irfs[i]
    return dl3_format_optional


class TestDL3GADFEventsWriter:
    def test_dl3_file(self, tmp_path, dl3_writer):
        output_path = tmp_path / "dl3_gadf.fits"

        dl3_writer.write_file(output_path)

        with fits.open(output_path, checksum=True) as hdul:
            assert isinstance(hdul[0], fits.PrimaryHDU)

            names = [hdu.name for hdu in hdul]
            assert "EVENTS" in names
            assert "GTI" in names
            assert "POINTING" in names

            irf_kinds = {
                hdu.header.get("HDUCLAS2")
                for hdu in hdul[1:]
                if "HDUCLAS2" in hdu.header
            }
            assert {"EFF_AREA", "EDISP", "PSF", "BKG"}.issubset(irf_kinds)

            for hdu in hdul:
                if "OBS_ID" in hdu.header:
                    assert hdu.header["OBS_ID"] == dl3_writer.obs_id

    def test_dl3_file_missing_aeff(self, tmp_path, dl3_writer):
        output_path = tmp_path / "dl3_gadf_aeff.fits"

        dl3_writer._aeff = None
        with pytest.raises(ValueError):
            dl3_writer.write_file(output_path)

    def test_dl3_file_missing_edisp(self, tmp_path, dl3_writer):
        output_path = tmp_path / "dl3_gadf_edisp.fits"

        dl3_writer._edisp = None
        with pytest.raises(ValueError):
            dl3_writer.write_file(output_path)

    def test_dl3_file_missing_psf(self, tmp_path, dl3_writer):
        output_path = tmp_path / "dl3_gadf_psf.fits"

        dl3_writer._psf = None
        with pytest.raises(ValueError):
            dl3_writer.write_file(output_path)

    def test_dl3_file_overwrite(self, tmp_path, dl3_writer):
        output_path = tmp_path / "dl3_gadf_overwrite.fits"

        dl3_writer.write_file(output_path)
        with pytest.raises(OSError):
            dl3_writer.write_file(output_path)

    def test_hdu_header_base(self, dl3_writer):
        header = dl3_writer.get_hdu_header_base_format()

        assert header["HDUCLASS"] == "GADF"
        assert header["HDUVERS"] == "v0.3"
        assert header["CREATOR"].startswith("ctapipe")

        file_time = datetime.fromisoformat(header["CREATED"])
        assert (datetime.now(UTC) - file_time) < timedelta(hours=1)

    def test_hdu_header_time(self, dl3_writer):
        header = dl3_writer.get_hdu_header_base_time()

        for key in [
            "MJDREFI",
            "MJDREFF",
            "TIMEUNIT",
            "TIMEREF",
            "TIMESYS",
            "TSTART",
            "TSTOP",
            "ONTIME",
            "LIVETIME",
            "DEADC",
            "TELAPSE",
            "DATE-OBS",
            "DATE-BEG",
            "DATE-AVG",
            "DATE-END",
        ]:
            assert key in header

        assert isinstance(header["MJDREFI"], int)
        assert header["MJDREFI"] == 58119
        assert isinstance(header["MJDREFF"], float)
        assert 0.0 <= header["MJDREFF"] < 1.0
        assert header["TIMEREF"] == "TOPOCENTER"
        assert header["TIMESYS"] == "TAI"
        assert header["TIMEUNIT"] == "s"

        assert header["TSTOP"] > header["TSTART"]

        assert header["DEADC"] <= 1
        assert header["LIVETIME"] == pytest.approx(header["ONTIME"] * header["DEADC"])
        assert header["LIVETIME"] <= header["TELAPSE"]
        assert header["TELAPSE"] == pytest.approx(header["TSTOP"] - header["TSTART"])

        ref_mjd = header["MJDREFI"] + header["MJDREFF"]
        tref = Time(ref_mjd, format="mjd", scale="tai")
        tstart = Time(header["DATE-BEG"], format="fits", scale="tai")
        tavg = Time(header["DATE-AVG"], format="fits", scale="tai")
        tstop = Time(header["DATE-END"], format="fits", scale="tai")
        assert (tstart - tref).to_value(u.s) == pytest.approx(
            header["TSTART"], rel=1e-6
        )
        assert (tstop - tref).to_value(u.s) == pytest.approx(header["TSTOP"], rel=1e-6)
        assert (tavg >= tstart) & (tavg <= tstop)

    def test_hdu_header_time_missing_gti(self, dl3_writer):
        dl3_writer._gti = None
        with pytest.raises(ValueError):
            dl3_writer.get_hdu_header_base_time()

    def test_hdu_header_time_missing_deadtime(self, dl3_writer):
        dl3_writer._livetime_fraction = None
        with pytest.raises(ValueError):
            dl3_writer.get_hdu_header_base_time()

    def test_livetime_fraction_setter_validation(self, dl3_writer):
        dl3_writer.livetime_fraction = 0.0
        assert dl3_writer.livetime_fraction == 0.0

        dl3_writer.livetime_fraction = 1.0
        assert dl3_writer.livetime_fraction == 1.0

        for invalid in (-1e-3, 1.001, np.nan, np.inf, -np.inf):
            with pytest.raises(ValueError):
                dl3_writer.livetime_fraction = invalid

        for invalid in ([0.5], "0.5", True):
            with pytest.raises(TypeError):
                dl3_writer.livetime_fraction = invalid

        dl3_writer.livetime_fraction = None
        assert dl3_writer.livetime_fraction is None

    def test_obs_id_setter_validation(self, dl3_writer):
        dl3_writer.obs_id = np.int64(1234)
        assert dl3_writer.obs_id == 1234

        with pytest.raises(ValueError):
            dl3_writer.obs_id = -1

        for invalid in (1.2, "1", True):
            with pytest.raises(TypeError):
                dl3_writer.obs_id = invalid

        dl3_writer.obs_id = None
        assert dl3_writer.obs_id is None

    def test_events_setter_validation(self, dl3_writer):
        table = Table(dl3_writer.events, copy=False)
        dl3_writer.events = table
        assert dl3_writer.events is table

        with pytest.raises(TypeError):
            dl3_writer.events = {"not": "a table"}

        dl3_writer.events = None
        assert dl3_writer.events is None

    def test_pointing_setter_validation(self, dl3_writer):
        with pytest.raises(TypeError):
            dl3_writer.pointing = "not-a-sequence"

        with pytest.raises(ValueError):
            dl3_writer.pointing = [(Time("2020-01-01T00:00:00", scale="tai"),)]

        with pytest.raises(TypeError):
            dl3_writer.pointing = [(Time("2020-01-01T00:00:00", scale="tai"), object())]

        dl3_writer.pointing = None
        assert dl3_writer.pointing is None

    def test_pointing_mode_setter_validation(self, dl3_writer):
        dl3_writer.pointing_mode = "track"
        assert dl3_writer.pointing_mode == "TRACK"

        dl3_writer.pointing_mode = " drift "
        assert dl3_writer.pointing_mode == "DRIFT"

        with pytest.raises(TypeError):
            dl3_writer.pointing_mode = 1

        with pytest.raises(ValueError):
            dl3_writer.pointing_mode = "WOBBLE"

    def test_gti_setter_validation(self, dl3_writer):
        with pytest.raises(TypeError):
            dl3_writer.gti = "not-a-sequence"

        with pytest.raises(ValueError):
            dl3_writer.gti = [(Time("2020-01-01T00:00:00", scale="tai"),)]

        dl3_writer.gti = None
        assert dl3_writer.gti is None

    def test_location_setter_validation(self, dl3_writer):
        with pytest.raises(TypeError):
            dl3_writer.location = "not-a-location"

        dl3_writer.location = None
        assert dl3_writer.location is None

    @pytest.mark.parametrize("setter", ["aeff", "psf", "edisp", "bkg"])
    def test_irf_setter_validation(self, dl3_writer, setter):
        with pytest.raises(TypeError):
            setattr(dl3_writer, setter, "not-an-hdu")

    def test_telescope_information_setter_validation(self, dl3_writer):
        with pytest.raises(TypeError):
            dl3_writer.telescope_information = "not-a-mapping"

        with pytest.raises(ValueError, match="missing keys"):
            dl3_writer.telescope_information = {"organisation": "CTAO"}

    def test_target_information_setter_validation(self, dl3_writer):
        with pytest.raises(TypeError):
            dl3_writer.target_information = "not-a-mapping"

        with pytest.raises(ValueError, match="missing keys"):
            dl3_writer.target_information = {"observer": "UNKNOWN"}

        with pytest.raises(TypeError):
            dl3_writer.target_information = {
                "observer": "UNKNOWN",
                "object_name": "UNKNOWN",
                "object_coordinate": object(),
            }

    def test_software_information_setter_validation(self, dl3_writer):
        with pytest.raises(TypeError):
            dl3_writer.software_information = "not-a-mapping"

        with pytest.raises(ValueError, match="missing keys"):
            dl3_writer.software_information = {"analysis_version": "ctapipe X"}

    def test_hdu_header_obs_info(self, dl3_writer, dl2_meta_for_dl3):
        obs_only = dl3_writer.get_hdu_header_base_observation_information(
            obs_id_only=True
        )
        assert obs_only["OBS_ID"] == dl3_writer.obs_id
        assert len(obs_only) == 1

        full_header = dl3_writer.get_hdu_header_base_observation_information(
            obs_id_only=False
        )
        assert full_header["OBS_ID"] == dl3_writer.obs_id
        target = dl2_meta_for_dl3["target"]
        assert full_header["OBSERVER"] == target["observer"]
        assert full_header["OBJECT"] == target["object_name"]

    def test_hdu_header_obs_info_missing_obs_id(self, dl3_writer):
        dl3_writer._obs_id = None
        with pytest.raises(ValueError):
            dl3_writer.get_hdu_header_base_observation_information(obs_id_only=True)
        with pytest.raises(ValueError):
            dl3_writer.get_hdu_header_base_observation_information(obs_id_only=False)

    def test_hdu_header_subarray_info(self, dl3_writer, dl2_meta_for_dl3):
        header = dl3_writer.get_hdu_header_base_subarray_information()

        tel_info = dl2_meta_for_dl3["telescope_information"]
        assert header["ORIGIN"] == tel_info["organisation"]
        assert header["TELESCOP"] == tel_info["array"]
        assert header["INSTRUME"] == tel_info["subarray"]
        assert header["TELLIST"] == str(tel_info["telescope_list"])
        assert header["N_TELS"] == len(tel_info["telescope_list"])

    def test_hdu_header_software_info(self, dl3_writer, dl2_meta_for_dl3):
        header = dl3_writer.get_hdu_header_base_software_information()
        soft = dl2_meta_for_dl3["software_version"]
        assert header["DST_VER"] == soft["dst_version"]
        assert header["ANA_VER"] == soft["analysis_version"]
        assert header["CAL_VER"] == soft["calibration_version"]

        dl3_writer._software_information = None
        header = dl3_writer.get_hdu_header_base_software_information()
        assert len(header) == 0

    def test_hdu_header_pointing(self, dl3_writer, dl2_meta_for_dl3):
        header = dl3_writer.get_hdu_header_base_pointing()

        assert header["RADESYS"] == "ICRS"
        assert header["RADECSYS"] == "ICRS"
        assert header["EQUINOX"] == 2000.0
        assert header["OBS_MODE"] == dl2_meta_for_dl3["pointing"]["pointing_mode"]

        for key in ["RA_PNT", "DEC_PNT", "ALT_PNT", "AZ_PNT"]:
            assert np.isfinite(header[key])

        loc = dl2_meta_for_dl3["location"]
        assert header["GEOLON"] == pytest.approx(loc.lon.to_value(u.deg))
        assert header["GEOLAT"] == pytest.approx(loc.lat.to_value(u.deg))
        assert header["ALTITUDE"] == pytest.approx(loc.height.to_value(u.m))
        assert header["OBSGEO-X"] == pytest.approx(loc.x.to_value(u.m))
        assert header["OBSGEO-Y"] == pytest.approx(loc.y.to_value(u.m))
        assert header["OBSGEO-Z"] == pytest.approx(loc.z.to_value(u.m))

    def test_hdu_header_pointing_track_mode_regression(self, dl3_writer):
        """Regression: TRACK mode must not fail when interpolating RA around 0/360."""
        dl3_writer.pointing_mode = "TRACK"
        header = dl3_writer.get_hdu_header_base_pointing()

        assert header["OBS_MODE"] == "POINTING"
        for key in ["RA_PNT", "DEC_PNT", "ALT_PNT", "AZ_PNT"]:
            assert np.isfinite(header[key])

    def test_hdu_header_pointing_drift_mode_regression(self, dl3_writer):
        """Regression: DRIFT mode interpolation must provide valid angular quantities."""
        dl3_writer.pointing_mode = "DRIFT"
        header = dl3_writer.get_hdu_header_base_pointing()

        assert header["OBS_MODE"] == "DRIFT"
        for key in ["RA_PNT", "DEC_PNT", "ALT_PNT", "AZ_PNT"]:
            assert np.isfinite(header[key])

    def test_hdu_header_pointing_missing_pointing(self, dl3_writer):
        dl3_writer._pointing = None
        with pytest.raises(ValueError):
            dl3_writer.get_hdu_header_base_pointing()

    def test_hdu_header_pointing_missing_pointing_mode(self, dl3_writer):
        dl3_writer._pointing_mode = None
        with pytest.raises(ValueError):
            dl3_writer.get_hdu_header_base_pointing()

    def test_hdu_header_pointing_missing_location(self, dl3_writer):
        dl3_writer._location = None
        with pytest.raises(ValueError):
            dl3_writer.get_hdu_header_base_pointing()

    def test_hdu_header_events_hdu(self, dl3_writer):
        header = dl3_writer.get_hdu_header_events()

        assert header["HDUCLASS"] == "GADF"
        assert header["HDUCLAS1"] == "EVENTS"
        # some representative keys from the different helper headers
        for key in [
            "HDUCLASS",
            "HDUDOC",
            "HDUVERS",
            "HDUCLAS1",
            "OBS_ID",
            "TSTART",
            "TSTOP",
            "ONTIME",
            "LIVETIME",
            "DEADC",
            "OBS_MODE",
            "RA_PNT",
            "DEC_PNT",
            "ALT_PNT",
            "AZ_PNT",
            "RADESYS",
            "RADECSYS",
            "EQUINOX",
            "ORIGIN",
            "TELESCOP",
            "INSTRUME",
            "CREATOR",
        ]:
            assert key in header

    def test_hdu_header_gti_hdu(self, dl3_writer):
        header = dl3_writer.get_hdu_header_gti()

        for key in [
            "MJDREFI",
            "MJDREFF",
            "TIMEUNIT",
            "TIMEREF",
            "TIMESYS",
            "TSTART",
            "TSTOP",
            "ONTIME",
            "LIVETIME",
            "TELAPSE",
            "DATE-OBS",
            "DATE-BEG",
            "DATE-AVG",
            "DATE-END",
        ]:
            assert key in header

        assert header["HDUCLASS"] == "GADF"
        assert header["HDUCLAS1"] == "GTI"

    def test_hdu_header_pointing_hdu(self, dl3_writer):
        header = dl3_writer.get_hdu_header_pointing()

        assert header["HDUCLASS"] == "GADF"
        assert header["HDUCLAS1"] == "POINTING"
        assert "TSTART" not in header
        assert "TSTOP" not in header
        # Time reference keywords must be present so the TIME column
        # can be interpreted correctly.
        for key in ["MJDREFI", "MJDREFF", "TIMEUNIT", "TIMESYS", "TIMEREF"]:
            assert key in header, f"Time reference keyword {key} missing from POINTING"
        assert header["TIMEREF"] == "TOPOCENTER"
        assert header["TIMESYS"] == "TAI"
        assert header["TIMEUNIT"] == "s"
        for key in ["RA_PNT", "DEC_PNT", "ALT_PNT", "AZ_PNT", "OBS_ID"]:
            assert key in header

    def test_column_renaming(self, dl3_writer):
        events = dl3_writer.events
        renamed = dl3_writer.transform_events_columns_for_gadf_format(events)

        assert renamed.colnames == ["EVENT_ID", "TIME", "RA", "DEC", "ENERGY"]
        assert len(renamed) == len(events)
        assert renamed["TIME"].unit.is_equivalent(u.s)
        assert renamed["TIME"].ndim == 1
        assert renamed["TIME"].dtype.kind == "f"
        assert np.all(np.isfinite(renamed["TIME"]))
        if len(renamed) > 1:
            np.testing.assert_allclose(
                np.diff(renamed["TIME"].to_value(u.s)),
                1.0,
            )

        bad_events = events.copy()
        bad_events.remove_column("reco_energy")
        with pytest.raises(ValueError, match="Required column reco_energy is missing"):
            dl3_writer.transform_events_columns_for_gadf_format(bad_events)

    def test_gti_table(self, dl3_writer, dl2_meta_for_dl3):
        gti_table = dl3_writer.create_gti_table()

        assert gti_table.colnames == ["START", "STOP"]
        assert len(gti_table) == len(dl2_meta_for_dl3["gti"])

    def test_pointing_table(self, dl3_writer):
        pointing_table = dl3_writer.create_pointing_table()

        assert pointing_table.colnames == [
            "TIME",
            "RA_PNT",
            "DEC_PNT",
            "ALT_PNT",
            "AZ_PNT",
        ]
        assert len(pointing_table) >= 1

        times = pointing_table["TIME"].to_value(u.s)
        assert np.all(np.diff(times) >= 0)

        assert np.all(
            (-90.0 <= pointing_table["DEC_PNT"].to_value(u.deg))
            & (pointing_table["DEC_PNT"].to_value(u.deg) <= 90.0)
        )
        assert np.all(
            (-90.0 <= pointing_table["ALT_PNT"].to_value(u.deg))
            & (pointing_table["ALT_PNT"].to_value(u.deg) <= 90.0)
        )
        assert np.all(np.isfinite(pointing_table["RA_PNT"].to_value(u.deg)))

    def test_pointing_table_missing_pointing(self, dl3_writer):
        dl3_writer._pointing = None
        with pytest.raises(ValueError):
            dl3_writer.create_pointing_table()

    def test_pointing_table_missing_location(self, dl3_writer):
        dl3_writer._location = None
        with pytest.raises(ValueError):
            dl3_writer.create_pointing_table()

    def test_gti_table_is_sorted(self, dl3_writer, dl2_meta_for_dl3):
        """Regression test: GTI table must be sorted by START (bug #1.3)."""
        original_gti = dl3_writer.gti

        # Build GTI intervals in reverse chronological order
        ref = Time("2020-06-01T00:00:00", scale="tai")
        reversed_gti = [
            (ref + 200 * u.s, ref + 300 * u.s),
            (ref + 100 * u.s, ref + 200 * u.s),
            (ref + 0 * u.s, ref + 100 * u.s),
        ]
        dl3_writer.gti = reversed_gti

        gti_table = dl3_writer.create_gti_table()
        start_values = gti_table["START"].to_value(u.s)
        assert np.all(np.diff(start_values) >= 0), (
            "GTI START column must be sorted in ascending order"
        )

        # Restore original GTI
        dl3_writer.gti = original_gti
