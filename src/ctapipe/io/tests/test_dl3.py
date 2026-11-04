from datetime import UTC, datetime, timedelta

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import ICRS, AltAz, SkyCoord
from astropy.io import fits
from astropy.table import Column, QTable, Table, vstack
from astropy.time import Time, TimeDelta

from ...containers import PointingMode
from ...core import QualityQuery
from ...io import TableLoader
from ...io.astropy_helpers import join_allow_empty
from ...io.dl2_tables_preprocessing import DL2EventPreprocessor
from ...version import version as ctapipe_version
from ..dl3 import DL3EventsData, DL3GADFEventsWriter


@pytest.fixture
def hdu_irfs(dummy_irf_file):
    with fits.open(dummy_irf_file, checksum=True) as hdus:
        yield hdus


@pytest.fixture(scope="session")
def dl2_meta_for_dl3(single_obs_gamma_diffuse_full_reco_file):
    with TableLoader(
        single_obs_gamma_diffuse_full_reco_file,
        dl2=True,
        observation_info=True,
        simulated=False,
    ) as loader:
        meta = {
            "location": loader.subarray.reference_location,
            "telescope_information": {
                "organisation": "CTAO",
                "array": "CTAO-North",
                "subarray": "4LST",
                "telescope_list": np.array(
                    loader.subarray.get_tel_ids(loader.subarray.tel)
                ),
            },
            "target": {
                "observer": "SuperObserver",
                "object_name": "Crab",
                "object_coordinate": SkyCoord(
                    ra=83.6331 * u.deg, dec=22.0145 * u.deg, frame="icrs"
                ),
            },
            "software_version": {
                "analysis_version": "ctapipe " + ctapipe_version,
                "calibration_version": "UNKNOWN",
                "dst_version": "UNKNOWN",
            },
            "livetime_fraction": 0.97,
        }

        obs_info = loader.read_observation_information()
        sched_info = loader.read_scheduling_blocks()
        obs_all_info = join_allow_empty(obs_info, sched_info, "sb_id", "inner")
        row = obs_all_info[0]
        meta["obs_id"] = int(row["obs_id"])

        start_time = Time(row["actual_start_time"]).tai
        stop_time = start_time + TimeDelta(obs_all_info["actual_duration"].quantity[0])
        meta["gti"] = [(start_time, stop_time)]

        pointing = AltAz(
            alt=obs_all_info["subarray_pointing_lat"].quantity[0],
            az=obs_all_info["subarray_pointing_lon"].quantity[0],
            location=meta["location"],
            obstime=start_time,
        )
        meta["pointing"] = {
            "pointing_mode": PointingMode(row["pointing_mode"]).name,
            "pointing_list": [(start_time, pointing), (stop_time, pointing)],
        }
        return meta


@pytest.fixture(scope="session")
def dl2_events_for_dl3(single_obs_gamma_diffuse_full_reco_file, dl2_meta_for_dl3):
    preprocessor = DL2EventPreprocessor(
        energy_reconstructor="ExtraTreesRegressor",
        geometry_reconstructor="HillasReconstructor",
        gammaness_classifier="ExtraTreesClassifier",
        apply_derived_columns=False,
        allow_unsupported_pointing_frames=True,
        output_table_schema=[
            Column(name="obs_id", dtype=np.uint64),
            Column(name="event_id", dtype=np.uint64),
            Column(name="reco_energy", unit=u.TeV),
            Column(name="reco_az", unit=u.deg),
            Column(name="reco_alt", unit=u.deg),
            Column(name="pointing_az", unit=u.deg),
            Column(name="pointing_alt", unit=u.deg),
            Column(name="gh_score", dtype=np.float64),
        ],
    )
    preprocessor.quality_query = QualityQuery(
        parent=preprocessor,
        quality_criteria=[
            (
                "multiplicity 4",
                "np.count_nonzero(HillasReconstructor_telescopes,axis=1) >= 4",
            ),
            ("valid classifier", "ExtraTreesClassifier_is_valid"),
            ("valid geom reco", "HillasReconstructor_is_valid"),
            ("valid energy reco", "ExtraTreesRegressor_is_valid"),
        ],
    )

    chunks = []
    with TableLoader(
        single_obs_gamma_diffuse_full_reco_file,
        dl2=True,
        simulated=False,
        observation_info=True,
    ) as loader:
        reader = loader.read_subarray_events_chunked(
            1000,
            dl2=True,
            simulated=False,
            observation_info=True,
        )
        for _, _, events in reader:
            selected = events[preprocessor.quality_query.get_table_mask(events)]
            if len(selected) == 0:
                continue
            chunks.append(preprocessor.normalise_column_names(selected))

    if len(chunks) == 0:
        raise ValueError("No events available for DL3 writer tests")

    events = QTable(vstack(chunks, join_type="exact", metadata_conflicts="silent"))
    events["time"] = (
        Time("2020-01-01T00:00:00", scale="tai") + np.arange(len(events)) * u.s
    )

    reco = SkyCoord(
        alt=events["reco_alt"],
        az=events["reco_az"],
        frame=AltAz(
            location=dl2_meta_for_dl3["location"],
            obstime=Time(events["time"]),
        ),
    )
    reco_icrs = reco.transform_to(ICRS())
    events["reco_ra"] = reco_icrs.ra.to(u.deg)
    events["reco_dec"] = reco_icrs.dec.to(u.deg)
    return events


@pytest.fixture
def dl3_data(dl2_events_for_dl3, dl2_meta_for_dl3, hdu_irfs):
    aeff = None
    psf = None
    edisp = None
    bkg = None

    for i in range(1, len(hdu_irfs)):
        if "HDUCLAS2" in hdu_irfs[i].header.keys():
            if hdu_irfs[i].header["HDUCLAS2"] == "EFF_AREA":
                if aeff is None:
                    aeff = hdu_irfs[i]
                elif "EXTNAME" in hdu_irfs[i].header and not (
                    "PROTONS" in hdu_irfs[i].header["EXTNAME"]
                    or "ELECTRONS" in hdu_irfs[i].header["EXTNAME"]
                ):
                    aeff = hdu_irfs[i]
            elif hdu_irfs[i].header["HDUCLAS2"] == "EDISP":
                edisp = hdu_irfs[i]
            elif hdu_irfs[i].header["HDUCLAS2"] == "PSF":
                psf = hdu_irfs[i]
            elif hdu_irfs[i].header["HDUCLAS2"] == "BKG":
                bkg = hdu_irfs[i]

    return DL3EventsData(
        events=dl2_events_for_dl3,
        obs_id=dl2_meta_for_dl3["obs_id"],
        pointing=dl2_meta_for_dl3["pointing"]["pointing_list"],
        pointing_mode=dl2_meta_for_dl3["pointing"]["pointing_mode"],
        gti=dl2_meta_for_dl3["gti"],
        livetime_fraction=dl2_meta_for_dl3["livetime_fraction"],
        location=dl2_meta_for_dl3["location"],
        telescope_information=dl2_meta_for_dl3["telescope_information"],
        target_information=dl2_meta_for_dl3["target"],
        software_information=dl2_meta_for_dl3["software_version"],
        aeff=aeff,
        psf=psf,
        edisp=edisp,
        bkg=bkg,
    )


@pytest.fixture
def dl3_writer():
    return DL3GADFEventsWriter()


class TestDL3GADFEventsWriter:
    def test_dl3_file(self, tmp_path, dl3_writer, dl3_data):
        output_path = tmp_path / "dl3_gadf.fits"

        dl3_writer.write_file(output_path, dl3_data)

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
                    assert hdu.header["OBS_ID"] == dl3_data.obs_id

    def test_dl3_file_missing_aeff(self, tmp_path, dl3_writer, dl3_data):
        output_path = tmp_path / "dl3_gadf_aeff.fits"

        object.__setattr__(dl3_data, "aeff", None)
        with pytest.raises(ValueError):
            dl3_writer.write_file(output_path, dl3_data)

    def test_dl3_file_missing_edisp(self, tmp_path, dl3_writer, dl3_data):
        output_path = tmp_path / "dl3_gadf_edisp.fits"

        object.__setattr__(dl3_data, "edisp", None)
        with pytest.raises(ValueError):
            dl3_writer.write_file(output_path, dl3_data)

    def test_dl3_file_missing_psf(self, tmp_path, dl3_writer, dl3_data):
        output_path = tmp_path / "dl3_gadf_psf.fits"

        object.__setattr__(dl3_data, "psf", None)
        with pytest.raises(ValueError):
            dl3_writer.write_file(output_path, dl3_data)

    def test_dl3_file_overwrite(self, tmp_path, dl3_writer, dl3_data):
        output_path = tmp_path / "dl3_gadf_overwrite.fits"

        dl3_writer.write_file(output_path, dl3_data)
        with pytest.raises(OSError):
            dl3_writer.write_file(output_path, dl3_data)

    def test_writer_reuse_does_not_leak_state(self, tmp_path, dl3_writer, dl3_data):
        first_path = tmp_path / "dl3_gadf_first.fits"
        second_path = tmp_path / "dl3_gadf_second.fits"
        original_irf_obs_ids = [
            hdu.header.get("OBS_ID")
            for hdu in (dl3_data.aeff, dl3_data.psf, dl3_data.edisp, dl3_data.bkg)
            if hdu is not None
        ]

        second_data = DL3EventsData(
            events=dl3_data.events,
            obs_id=dl3_data.obs_id + 1,
            pointing=dl3_data.pointing,
            pointing_mode=dl3_data.pointing_mode,
            gti=dl3_data.gti,
            livetime_fraction=dl3_data.livetime_fraction,
            location=dl3_data.location,
            telescope_information=dl3_data.telescope_information,
            target_information=dl3_data.target_information,
            software_information=dl3_data.software_information,
            aeff=dl3_data.aeff,
            psf=dl3_data.psf,
            edisp=dl3_data.edisp,
            bkg=dl3_data.bkg,
        )

        dl3_writer.write_file(first_path, dl3_data)
        dl3_writer.write_file(second_path, second_data)

        with fits.open(first_path, checksum=True) as first_hdul:
            assert first_hdul["EVENTS"].header["OBS_ID"] == dl3_data.obs_id
        with fits.open(second_path, checksum=True) as second_hdul:
            assert second_hdul["EVENTS"].header["OBS_ID"] == second_data.obs_id

        current_irf_obs_ids = [
            hdu.header.get("OBS_ID")
            for hdu in (dl3_data.aeff, dl3_data.psf, dl3_data.edisp, dl3_data.bkg)
            if hdu is not None
        ]
        assert current_irf_obs_ids == original_irf_obs_ids

    def test_hdu_header_base(self, dl3_writer, dl3_data):
        header = dl3_writer.get_hdu_header_base_format()

        assert header["HDUCLASS"] == "GADF"
        assert header["HDUVERS"] == "v0.3"
        assert header["CREATOR"].startswith("ctapipe")

        file_time = datetime.fromisoformat(header["CREATED"])
        assert (datetime.now(UTC) - file_time) < timedelta(hours=1)

    def test_hdu_header_time(self, dl3_writer, dl3_data):
        header = dl3_writer.get_hdu_header_base_time(dl3_data)

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

    def test_hdu_header_time_missing_gti(self, dl3_writer, dl3_data):
        object.__setattr__(dl3_data, "gti", None)
        with pytest.raises(ValueError):
            dl3_writer.get_hdu_header_base_time(dl3_data)

    def test_hdu_header_time_missing_deadtime(self, dl3_writer, dl3_data):
        object.__setattr__(dl3_data, "livetime_fraction", None)
        with pytest.raises(ValueError):
            dl3_writer.get_hdu_header_base_time(dl3_data)

    def test_livetime_fraction_setter_validation(self, dl3_writer, dl3_data):
        dl3_data.livetime_fraction = 0.0
        assert dl3_data.livetime_fraction == 0.0

        dl3_data.livetime_fraction = 1.0
        assert dl3_data.livetime_fraction == 1.0

        for invalid in (-1e-3, 1.001, np.nan, np.inf, -np.inf):
            with pytest.raises(ValueError):
                dl3_data.livetime_fraction = invalid

        for invalid in ([0.5], "0.5", True):
            with pytest.raises(TypeError):
                dl3_data.livetime_fraction = invalid

        with pytest.raises(ValueError):
            dl3_data.livetime_fraction = None

    def test_obs_id_setter_validation(self, dl3_writer, dl3_data):
        dl3_data.obs_id = np.int64(1234)
        assert dl3_data.obs_id == 1234

        with pytest.raises(ValueError):
            dl3_data.obs_id = -1

        for invalid in (1.2, "1", True):
            with pytest.raises(TypeError):
                dl3_data.obs_id = invalid

        with pytest.raises(ValueError):
            dl3_data.obs_id = None

    def test_events_setter_validation(self, dl3_writer, dl3_data):
        qtable = QTable(dl3_data.events, copy=True)
        dl3_data.events = qtable
        assert dl3_data.events is qtable

        table = Table(dl3_data.events, copy=False)
        with pytest.raises(TypeError):
            dl3_data.events = table

        with pytest.raises(TypeError):
            dl3_data.events = {"not": "a table"}

        with pytest.raises(ValueError):
            dl3_data.events = None

    def test_pointing_setter_validation(self, dl3_writer, dl3_data):
        with pytest.raises(TypeError):
            dl3_data.pointing = "not-a-sequence"

        with pytest.raises(ValueError):
            dl3_data.pointing = [(Time("2020-01-01T00:00:00", scale="tai"),)]

        with pytest.raises(TypeError):
            dl3_data.pointing = [(Time("2020-01-01T00:00:00", scale="tai"), object())]

        with pytest.raises(ValueError):
            dl3_data.pointing = None

    def test_pointing_mode_setter_validation(self, dl3_writer, dl3_data):
        dl3_data.pointing_mode = "track"
        assert dl3_data.pointing_mode == "TRACK"

        dl3_data.pointing_mode = " drift "
        assert dl3_data.pointing_mode == "DRIFT"

        with pytest.raises(TypeError):
            dl3_data.pointing_mode = 1

        with pytest.raises(ValueError):
            dl3_data.pointing_mode = "WOBBLE"

    def test_gti_setter_validation(self, dl3_writer, dl3_data):
        with pytest.raises(TypeError):
            dl3_data.gti = "not-a-sequence"

        with pytest.raises(ValueError):
            dl3_data.gti = [(Time("2020-01-01T00:00:00", scale="tai"),)]

        with pytest.raises(ValueError):
            dl3_data.gti = None

    def test_location_setter_validation(self, dl3_writer, dl3_data):
        with pytest.raises(TypeError):
            dl3_data.location = "not-a-location"

        with pytest.raises(ValueError):
            dl3_data.location = None

    @pytest.mark.parametrize("setter", ["aeff", "psf", "edisp", "bkg"])
    def test_irf_setter_validation(self, dl3_writer, dl3_data, setter):
        with pytest.raises(TypeError):
            setattr(dl3_data, setter, "not-an-hdu")

    def test_telescope_information_setter_validation(self, dl3_writer, dl3_data):
        with pytest.raises(TypeError):
            dl3_data.telescope_information = "not-a-mapping"

        with pytest.raises(ValueError, match="missing keys"):
            dl3_data.telescope_information = {"organisation": "CTAO"}

    def test_target_information_setter_validation(self, dl3_writer, dl3_data):
        with pytest.raises(TypeError):
            dl3_data.target_information = "not-a-mapping"

        with pytest.raises(ValueError, match="missing keys"):
            dl3_data.target_information = {"observer": "UNKNOWN"}

        with pytest.raises(TypeError):
            dl3_data.target_information = {
                "observer": "UNKNOWN",
                "object_name": "UNKNOWN",
                "object_coordinate": object(),
            }

    def test_software_information_setter_validation(self, dl3_writer, dl3_data):
        with pytest.raises(TypeError):
            dl3_data.software_information = "not-a-mapping"

        with pytest.raises(ValueError, match="missing keys"):
            dl3_data.software_information = {"analysis_version": "ctapipe X"}

    def test_hdu_header_obs_info(self, dl3_writer, dl3_data, dl2_meta_for_dl3):
        obs_only = dl3_writer.get_hdu_header_base_observation_information(
            dl3_data, obs_id_only=True
        )
        assert obs_only["OBS_ID"] == dl3_data.obs_id
        assert len(obs_only) == 1

        full_header = dl3_writer.get_hdu_header_base_observation_information(
            dl3_data, obs_id_only=False
        )
        assert full_header["OBS_ID"] == dl3_data.obs_id
        target = dl2_meta_for_dl3["target"]
        assert full_header["OBSERVER"] == target["observer"]
        assert full_header["OBJECT"] == target["object_name"]

    def test_hdu_header_obs_info_missing_obs_id(self, dl3_writer, dl3_data):
        object.__setattr__(dl3_data, "obs_id", None)
        with pytest.raises(ValueError):
            dl3_writer.get_hdu_header_base_observation_information(
                dl3_data, obs_id_only=True
            )
        with pytest.raises(ValueError):
            dl3_writer.get_hdu_header_base_observation_information(
                dl3_data, obs_id_only=False
            )

    def test_hdu_header_subarray_info(self, dl3_writer, dl3_data, dl2_meta_for_dl3):
        header = dl3_writer.get_hdu_header_base_subarray_information(dl3_data)

        tel_info = dl2_meta_for_dl3["telescope_information"]
        assert header["ORIGIN"] == tel_info["organisation"]
        assert header["TELESCOP"] == tel_info["array"]
        assert header["INSTRUME"] == tel_info["subarray"]
        assert header["TELLIST"] == str(tel_info["telescope_list"])
        assert header["N_TELS"] == len(tel_info["telescope_list"])

    def test_hdu_header_software_info(self, dl3_writer, dl3_data, dl2_meta_for_dl3):
        header = dl3_writer.get_hdu_header_base_software_information(dl3_data)
        soft = dl2_meta_for_dl3["software_version"]
        assert header["DST_VER"] == soft["dst_version"]
        assert header["ANA_VER"] == soft["analysis_version"]
        assert header["CAL_VER"] == soft["calibration_version"]

        object.__setattr__(dl3_data, "software_information", None)
        header = dl3_writer.get_hdu_header_base_software_information(dl3_data)
        assert len(header) == 0

    def test_hdu_header_pointing(self, dl3_writer, dl3_data, dl2_meta_for_dl3):
        header = dl3_writer.get_hdu_header_base_pointing(dl3_data)

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

    def test_hdu_header_pointing_track_mode_regression(self, dl3_writer, dl3_data):
        dl3_data.pointing_mode = "TRACK"
        header = dl3_writer.get_hdu_header_base_pointing(dl3_data)

        assert header["OBS_MODE"] == "POINTING"
        for key in ["RA_PNT", "DEC_PNT", "ALT_PNT", "AZ_PNT"]:
            assert np.isfinite(header[key])

    def test_hdu_header_pointing_drift_mode_regression(self, dl3_writer, dl3_data):
        dl3_data.pointing_mode = "DRIFT"
        header = dl3_writer.get_hdu_header_base_pointing(dl3_data)

        assert header["OBS_MODE"] == "DRIFT"
        for key in ["RA_PNT", "DEC_PNT", "ALT_PNT", "AZ_PNT"]:
            assert np.isfinite(header[key])

    def test_hdu_header_pointing_missing_pointing(self, dl3_writer, dl3_data):
        object.__setattr__(dl3_data, "pointing", None)
        with pytest.raises(ValueError):
            dl3_writer.get_hdu_header_base_pointing(dl3_data)

    def test_hdu_header_pointing_missing_pointing_mode(self, dl3_writer, dl3_data):
        object.__setattr__(dl3_data, "pointing_mode", None)
        with pytest.raises(ValueError):
            dl3_writer.get_hdu_header_base_pointing(dl3_data)

    def test_hdu_header_pointing_missing_location(self, dl3_writer, dl3_data):
        object.__setattr__(dl3_data, "location", None)
        with pytest.raises(ValueError):
            dl3_writer.get_hdu_header_base_pointing(dl3_data)

    def test_hdu_header_events_hdu(self, dl3_writer, dl3_data):
        header = dl3_writer.get_hdu_header_events(dl3_data)

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

    def test_hdu_header_gti_hdu(self, dl3_writer, dl3_data):
        header = dl3_writer.get_hdu_header_gti(dl3_data)

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

    def test_hdu_header_pointing_hdu(self, dl3_writer, dl3_data):
        header = dl3_writer.get_hdu_header_pointing(dl3_data)

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

    def test_column_renaming(self, dl3_writer, dl3_data):
        events = dl3_data.events
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

    def test_gti_table(self, dl3_writer, dl3_data, dl2_meta_for_dl3):
        gti_table = dl3_writer.create_gti_table(dl3_data)

        assert gti_table.colnames == ["START", "STOP"]
        assert len(gti_table) == len(dl2_meta_for_dl3["gti"])

    def test_pointing_table(self, dl3_writer, dl3_data):
        pointing_table = dl3_writer.create_pointing_table(dl3_data)

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

    def test_pointing_table_missing_pointing(self, dl3_writer, dl3_data):
        object.__setattr__(dl3_data, "pointing", None)
        with pytest.raises(ValueError):
            dl3_writer.create_pointing_table(dl3_data)

    def test_pointing_table_missing_location(self, dl3_writer, dl3_data):
        object.__setattr__(dl3_data, "location", None)
        with pytest.raises(ValueError):
            dl3_writer.create_pointing_table(dl3_data)

    def test_gti_table_is_sorted(self, dl3_writer, dl3_data, dl2_meta_for_dl3):
        """Regression test: GTI table must be sorted by START (bug #1.3)."""
        original_gti = dl3_data.gti

        # Build GTI intervals in reverse chronological order
        ref = Time("2020-06-01T00:00:00", scale="tai")
        reversed_gti = [
            (ref + 200 * u.s, ref + 300 * u.s),
            (ref + 100 * u.s, ref + 200 * u.s),
            (ref + 0 * u.s, ref + 100 * u.s),
        ]
        dl3_data.gti = reversed_gti

        gti_table = dl3_writer.create_gti_table(dl3_data)
        start_values = gti_table["START"].to_value(u.s)
        assert np.all(np.diff(start_values) >= 0), (
            "GTI START column must be sorted in ascending order"
        )

        # Restore original GTI
        dl3_data.gti = original_gti
