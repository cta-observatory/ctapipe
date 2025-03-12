from abc import abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Tuple

import astropy.units as u
import numpy as np
from astropy.coordinates import ICRS, EarthLocation, SkyCoord
from astropy.io import fits
from astropy.io.fits import Header
from astropy.io.fits.hdu.base import ExtensionHDU
from astropy.table import QTable
from astropy.time import Time, TimeDelta

from ctapipe.compat import COPY_IF_NEEDED
from ctapipe.core import Component
from ctapipe.core.traits import Bool
from ctapipe.version import version as ctapipe_version


class DL3_Format(Component):
    overwrite = Bool(
        default_value=False,
        help="If true, allow to overwrite already existing output file",
    ).tag(config=True)

    optional_dl3_columns = Bool(
        default_value=False, help="If true add optional columns to produce file"
    ).tag(config=False)

    raise_error_for_optional = Bool(
        default_value=True,
        help="If true will raise error in the case optional column are missing",
    ).tag(config=False)

    raise_error_for_missing_hdu = Bool(
        default_value=True,
        help="If true will raise error if HDU are missing from the final DL3 file",
    ).tag(config=False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._obs_id = None
        self._events = None
        self._pointing = None
        self._gti = None
        self._aeff = None
        self._psf = None
        self._edisp = None
        self._bkg = None
        self._dead_time_fraction = None
        self._location = None
        self._telescope_information = None
        self._target_information = None
        self._software_information = None

    @abstractmethod
    def write_file(self, path):
        pass

    @property
    def obs_id(self) -> int:
        return self._obs_id

    @obs_id.setter
    def obs_id(self, obs_id: int):
        if self._obs_id is not None:
            self.log.warning(
                "Obs id for DL3 file was already set, replacing current obs id"
            )
        self._obs_id = obs_id

    @property
    def events(self) -> QTable:
        return self._events

    @events.setter
    def events(self, events: QTable):
        if self._events is not None:
            self.log.warning(
                "Events table for DL3 file was already set, replacing current event table"
            )
        self._events = events

    @property
    def pointing(self) -> List[Tuple[Time, SkyCoord]]:
        return self._pointing

    @pointing.setter
    def pointing(self, pointing: List[Tuple[Time, SkyCoord]]):
        if self._pointing is not None:
            self.log.warning(
                "Pointing for DL3 file was already set, replacing current pointing"
            )
        self._pointing = pointing

    @property
    def gti(self) -> List[Tuple[Time, Time]]:
        return self._gti

    @gti.setter
    def gti(self, gti: List[Tuple[Time, Time]]):
        if self._gti is not None:
            self.log.warning("GTI for DL3 file was already set, replacing current gti")
        self._gti = gti

    @property
    def aeff(self) -> ExtensionHDU:
        return self._aeff

    @aeff.setter
    def aeff(self, aeff: ExtensionHDU):
        if self._aeff is not None:
            self.log.warning(
                "Effective area for DL3 file was already set, replacing current effective area"
            )
        self._aeff = aeff

    @property
    def psf(self) -> ExtensionHDU:
        return self._psf

    @psf.setter
    def psf(self, psf: ExtensionHDU):
        if self._psf is not None:
            self.log.warning("PSF for DL3 file was already set, replacing current PSF")
        self._psf = psf

    @property
    def edisp(self) -> ExtensionHDU:
        return self._edisp

    @edisp.setter
    def edisp(self, edisp: ExtensionHDU):
        if self._edisp is not None:
            self.log.warning(
                "EDISP for DL3 file was already set, replacing current EDISP"
            )
        self._edisp = edisp

    @property
    def bkg(self) -> ExtensionHDU:
        return self._bkg

    @bkg.setter
    def bkg(self, bkg: ExtensionHDU):
        if self._bkg is not None:
            self.log.warning(
                "Background for DL3 file was already set, replacing current background"
            )
        self._bkg = bkg

    @property
    def location(self) -> EarthLocation:
        return self._location

    @location.setter
    def location(self, location: EarthLocation):
        if self._location is not None:
            self.log.warning(
                "Telescope location for DL3 file was already set, replacing current location"
            )
        self._location = location

    @property
    def dead_time_fraction(self) -> float:
        return self._dead_time_fraction

    @dead_time_fraction.setter
    def dead_time_fraction(self, dead_time_fraction: float):
        if self.dead_time_fraction is not None:
            self.log.warning(
                "Dead time fraction for DL3 file was already set, replacing current dead time fraction"
            )
        self._dead_time_fraction = dead_time_fraction

    @property
    def telescope_information(self) -> Dict[str, Any]:
        return self._telescope_information

    @telescope_information.setter
    def telescope_information(self, telescope_information: Dict[str, Any]):
        if self._telescope_information is not None:
            self.log.warning(
                "Telescope information for DL3 file was already set, replacing current information"
            )
        self._telescope_information = telescope_information

    @property
    def target_information(self) -> Dict[str, Any]:
        return self._target_information

    @target_information.setter
    def target_information(self, target_information: Dict[str, Any]):
        if self._target_information is not None:
            self.log.warning(
                "Target information for DL3 file was already set, replacing current target information"
            )
        self._target_information = target_information

    @property
    def software_information(self) -> Dict[str, Any]:
        return self._software_information

    @software_information.setter
    def software_information(self, software_information: Dict[str, Any]):
        if self._software_information is not None:
            self.log.warning(
                "Software information for DL3 file was already set, replacing current software information"
            )
        self._software_information = software_information


class DL3_GADF(DL3_Format):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.file_creation_time = datetime.now(tz=datetime.UTC).isoformat()
        self.reference_time = Time(datetime.fromisoformat("1970-01-01T00:00:00+00:00"))

    def write_file(self, path):
        self.file_creation_time = datetime.now(tz=datetime.UTC).isoformat()

        hdu_dl3 = fits.HDUList(
            [fits.PrimaryHDU(header=Header(self.get_hdu_header_base_format()))]
        )
        hdu_dl3.append(
            fits.BinTableHDU(
                data=self.transform_events_columns_for_gadf_format(self.events),
                name="EVENTS",
                header=Header(self.get_hdu_header_events()),
            )
        )
        hdu_dl3.append(
            fits.BinTableHDU(
                data=self.create_gti_table(),
                name="GTI",
                header=Header(self.get_hdu_header_gti()),
            )
        )
        hdu_dl3.append(self.aeff)
        hdu_dl3.append(self.psf)
        hdu_dl3.append(self.edisp)
        hdu_dl3.append(self.bkg)

        hdu_dl3.writeto(path, checksum=True, overwrite=self.overwrite)

    def get_hdu_header_base_format(self):
        return {
            "HDUCLASS": "GADF",
            "HDUVERS": "v0.3",
            "HDUDOC": "https://gamma-astro-data-formats.readthedocs.io/en/v0.3/index.html",
            "CREATOR": "ctapipe " + ctapipe_version,
            "CREATED": self.file_creation_time,
        }

    def get_hdu_header_base_time(self):
        if self.gti is None:
            raise ValueError("No available time information for the DL3 file")
        if self.dead_time_fraction is None:
            raise ValueError("No available dead time fraction for the DL3 file")
        start_time = None
        stop_time = None
        ontime = TimeDelta(0.0 * u.s)
        for gti_interval in self.gti:
            ontime += gti_interval[1] - gti_interval[0]
            start_time = (
                gti_interval[0]
                if start_time is None
                else min(start_time, gti_interval[0])
            )
            stop_time = (
                gti_interval[1]
                if stop_time is None
                else max(stop_time, gti_interval[1])
            )

        return {
            "MJDREFI": int(self.reference_time.mjd),
            "MJDREFF": self.reference_time.mjd % 1,
            "TIMEUNIT": "s",
            "TIMEREF": "GEOCENTER",
            "TIMESYS": "UTC",
            "TSTART": start_time,
            "TSTOP": stop_time,
            "ONTIME": ontime.to_value(u.s),
            "LIVETIME": ontime.to_value(u.s) * self.dead_time_fraction,
            "DEADC": self.dead_time_fraction,
            "TELAPSE": (stop_time - start_time).to_value(u.s),
            "DATE-OBS": start_time.fits,
            "DATE-BEG": start_time.fits,
            "DATE-AVG": (start_time + (stop_time - start_time) / 2.0).fits,
            "DATE-END": stop_time.fits,
        }

    def get_hdu_header_observation_information(self, obs_id_only=False):
        if self.obs_id is None:
            raise ValueError("Observation ID is missing.")
        header = {"OBS_ID": self.obs_id}
        if self.target_information is not None and not obs_id_only:
            header["OBSERVER"] = self.target_information["observer"]
            header["OBJECT"] = self.target_information["object_name"]
            object_coordinate = self.target_information[
                "object_coordinate"
            ].transform_to(ICRS())
            header["RA_OBJ"] = object_coordinate.ra.to_value(u.deg)
            header["DEC_OBJ"] = object_coordinate.dec.to_value(u.deg)
        return header

    def get_hdu_header_subarray_information(self):
        if self.telescope_information is None:
            raise ValueError("Telescope information are missing.")
        header = {
            "ORIGIN": self.telescope_information["organisation"],
            "TELESCOP": self.telescope_information["array"],
            "INSTRUME": self.telescope_information["subarray"],
            "TELLIST": self.telescope_information["telescope_list"],
            "N_TELS": np.sum(self.telescope_information["telescope_list"]),
        }
        return header

    def get_hdu_header_software_information(self):
        header = {}
        if self.software_information is not None:
            header["DST_VER"] = self.software_information["dst_version"]
            header["ANA_VER"] = self.software_information["analysis_version"]
            header["CAL_VER"] = self.software_information["calibration_version"]
        return header

    def get_hdu_header_events(self):
        header = self.get_hdu_header_base_format()
        header.update({"HDUCLAS1": "EVENTS"})
        header.update(self.get_hdu_header_base_time())
        header.update(self.get_hdu_header_observation_information())
        header.update(self.get_hdu_header_subarray_information())
        header.update(self.get_hdu_header_software_information())
        return header

    def get_hdu_header_gti(self):
        header = self.get_hdu_header_base_format()
        header.update({"HDUCLAS1": "GTI"})
        header.update(self.get_hdu_header_base_time())
        header.update(self.get_hdu_header_observation_information(obs_id_only=True))
        return header

    def transform_events_columns_for_gadf_format(self, events):
        rename_from = ["event_id", "time", "reco_ra", "reco_dec", "reco_energy"]
        rename_to = ["EVENT_ID", "TIME", "RA", "DEC", "ENERGY"]

        if self.optional_dl3_columns:
            rename_from_optional = [
                "multiplicity",
                "reco_glon",
                "reco_glat",
                "reco_alt",
                "reco_az",
                "reco_fov_lon",
                "reco_fov_lat",
                "reco_source_fov_offset",
                "reco_source_fov_position_angle",
                "gh_score",
                "reco_dir_uncert",
                "reco_energy_uncert",
                "reco_core_x",
                "reco_core_y",
                "reco_core_uncert",
                "reco_h_max",
                "reco_h_max_uncert",
            ]
            rename_to_optional = [
                "MULTIP",
                "GLON",
                "GLAT",
                "ALT",
                "AZ",
                "DETX",
                "DETY",
                "THETA",
                "PHI",
                "GAMANESS",
                "DIR_ERR",
                "ENERGY_ERR",
                "COREX",
                "COREY",
                "CORE_ERR",
                "HMAX",
                "HMAX_ERR",
            ]

            if not self.raise_error_for_optional:
                for i, c in enumerate(rename_from_optional):
                    if c not in events.colnames:
                        self.log.warning(
                            f"Optional column {c} is missing from the events table."
                        )
                    else:
                        rename_from.append(rename_from_optional[i])
                        rename_to.append(rename_to_optional[i])

        for c in rename_from:
            if c not in events.colnames:
                raise ValueError(
                    f"Required column {c} is missing from the events table."
                )

        renamed_events = QTable(events, copy=COPY_IF_NEEDED)
        renamed_events.rename_columns(rename_from, rename_to)
        renamed_events = renamed_events[rename_to]
        return renamed_events

    def create_gti_table(self) -> QTable:
        table_structure = {"START": [], "STOP": []}
        for gti_interval in self.gti:
            table_structure["START"].append(
                (gti_interval[0] - self.reference_time).to(u.s)
            )
            table_structure["STOP"].append(
                (gti_interval[1] - self.reference_time).to(u.s)
            )

        table = QTable(table_structure).sort("START")
        for i in range(len(QTable) - 1):
            if table_structure["STOP"][i] > table_structure["START"][i + 1]:
                self.log.warning("Overlapping GTI intervals")
                break

        return table
