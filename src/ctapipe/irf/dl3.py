from abc import abstractmethod
from datetime import UTC, datetime
from typing import Any, Dict, List, Tuple

import astropy.units as u
import numpy as np
from astropy.coordinates import ICRS, AltAz, EarthLocation, SkyCoord
from astropy.io import fits
from astropy.io.fits import Header
from astropy.io.fits.hdu.base import ExtensionHDU
from astropy.table import QTable
from astropy.time import Time, TimeDelta

from ctapipe.compat import COPY_IF_NEEDED
from ctapipe.core import Component
from ctapipe.core.traits import AstroTime, Bool
from ctapipe.version import version as ctapipe_version


class DL3_Format(Component):
    """
    Base class for writing a DL3 file
    """

    overwrite = Bool(
        default_value=False,
        help="If true, allow to overwrite already existing output file",
    ).tag(config=True)

    optional_dl3_columns = Bool(
        default_value=False, help="If true add optional columns to produce file"
    ).tag(config=True)

    raise_error_for_optional = Bool(
        default_value=True,
        help="If true will raise error in the case optional column are missing",
    ).tag(config=True)

    reference_time = AstroTime(
        default_value=Time("2018-01-01T00:00:00", scale="tai"),
        help="The reference time that will be used in the FITS file",
    ).tag(config=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._obs_id = None
        self._events = None
        self._pointing = None
        self._pointing_mode = None
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
        """
        Parameters
        ----------
        obs_id : int
            Observation ID
        """
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
        """
        Parameters
        ----------
        events : QTable
            A table with a line for each event
        """
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
        """
        Parameters
        ----------
        pointing : List[Tuple[Time, SkyCoord]]
            A list with for each entry containing the time at which the coordinate where evaluated and the associated coordinates
        """
        if self._pointing is not None:
            self.log.warning(
                "Pointing for DL3 file was already set, replacing current pointing"
            )
        self._pointing = pointing

    @property
    def pointing_mode(self) -> str:
        return self._pointing_mode

    @pointing_mode.setter
    def pointing_mode(self, pointing_mode: str):
        """
        Parameters
        ----------
        pointing_mode : str
            The name of the pointing mode used for the observation
        """
        if self._pointing_mode is not None:
            self.log.warning(
                "Pointing for DL3 file was already set, replacing current pointing"
            )
        self._pointing_mode = pointing_mode

    @property
    def gti(self) -> List[Tuple[Time, Time]]:
        return self._gti

    @gti.setter
    def gti(self, gti: List[Tuple[Time, Time]]):
        """
        Parameters
        ----------
        gti : List[Tuple[Time, Time]]
            A list with for each entry containing the time at which the coordinate where evaluated and
        """
        if self._gti is not None:
            self.log.warning("GTI for DL3 file was already set, replacing current gti")
        self._gti = gti

    @property
    def aeff(self) -> ExtensionHDU:
        return self._aeff

    @aeff.setter
    def aeff(self, aeff: ExtensionHDU):
        """
        Parameters
        ----------
        aeff : ExtensionHDU
            The effective area HDU read from the fits file containing IRFs
        """
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
        """
        Parameters
        ----------
        psf : ExtensionHDU
            The PSF HDU read from the fits file containing IRFs
        """
        if self._psf is not None:
            self.log.warning("PSF for DL3 file was already set, replacing current PSF")
        self._psf = psf

    @property
    def edisp(self) -> ExtensionHDU:
        return self._edisp

    @edisp.setter
    def edisp(self, edisp: ExtensionHDU):
        """
        Parameters
        ----------
        edisp : ExtensionHDU
            The EDISP HDU read from the fits file containing IRFs
        """
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
        """
        Parameters
        ----------
        bkg : ExtensionHDU
            The background HDU read from the fits file containing IRFs
        """
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
        """
        Parameters
        ----------
        location : EarthLocation
            The location of the telescope
        """
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
        """
        Parameters
        ----------
        dead_time_fraction : float
            The dead time fraction fir the observations
        """
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
        """
        Parameters
        ----------
        telescope_information : dict[str, any]
            A dictionary containing general information about telescope with as key : organisation, array, subarray, telescope_list
        """
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
        """
        Parameters
        ----------
        target_information : dict[str, any]
            A dictionary containing general information about the targeted source with as key : observer, object_name, object_coordinate
        """
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
        """
        Parameters
        ----------
        software_information : dict[str, any]
            A dictionary containing general information about the software used to produce the file with as key : analysis_version, calibration_version, dst_version
        """
        if self._software_information is not None:
            self.log.warning(
                "Software information for DL3 file was already set, replacing current software information"
            )
        self._software_information = software_information


class DL3_GADF(DL3_Format):
    """
    Class to write DL3 in GADF format, subclass of DL3_Format
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.file_creation_time = datetime.now(tz=UTC)
        self._reference_time = self.reference_time.tai

    def write_file(self, path):
        """
        This function will write the new DL3 file
        All the content associated with the file should have been specified previously, otherwise error will be raised

        Parameters
        ----------
        path : str
            The full path and filename of the new file to write
        """
        self.file_creation_time = datetime.now(tz=UTC)

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
        hdu_dl3.append(
            fits.BinTableHDU(
                data=self.create_pointing_table(),
                name="POINTING",
                header=Header(self.get_hdu_header_pointing()),
            )
        )

        if self.aeff is None:
            raise ValueError("Missing effective area IRF")
        hdu_dl3.append(self.aeff)
        hdu_dl3[-1].header["OBS_ID"] = self.obs_id
        if self.psf is None:
            raise ValueError("Missing PSF IRF")
        hdu_dl3.append(self.psf)
        hdu_dl3[-1].header["OBS_ID"] = self.obs_id
        if self.edisp is None:
            raise ValueError("Missing EDISP IRF")
        hdu_dl3.append(self.edisp)
        hdu_dl3[-1].header["OBS_ID"] = self.obs_id
        if self.bkg is not None:
            hdu_dl3.append(self.bkg)
            hdu_dl3[-1].header["OBS_ID"] = self.obs_id

        hdu_dl3.writeto(path, checksum=True, overwrite=self.overwrite)

    def get_hdu_header_base_format(self) -> Dict[str, Any]:
        """
        Return the base information that should be included in all HDU of the final fits file
        """
        return {
            "HDUCLASS": "GADF",
            "HDUVERS": "v0.3",
            "HDUDOC": "https://gamma-astro-data-formats.readthedocs.io/en/v0.3/index.html",
            "CREATOR": "ctapipe " + ctapipe_version,
            "CREATED": self.file_creation_time.isoformat(),
        }

    def get_hdu_header_base_time(self) -> Dict[str, Any]:
        """
        Return the information about time parameters used in several HDU
        """
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
            "MJDREFI": int(self._reference_time.mjd),
            "MJDREFF": self._reference_time.mjd % 1,
            "TIMEUNIT": "s",
            "TIMEREF": "GEOCENTER",
            "TIMESYS": "TAI",
            "TSTART": (start_time.tai - self._reference_time).to_value(u.s),
            "TSTOP": (stop_time.tai - self._reference_time).to_value(u.s),
            "ONTIME": ontime.to_value(u.s),
            "LIVETIME": ontime.to_value(u.s) * self.dead_time_fraction,
            "DEADC": self.dead_time_fraction,
            "TELAPSE": (stop_time.tai - start_time.tai).to_value(u.s),
            "DATE-OBS": start_time.tai.fits,
            "DATE-BEG": start_time.tai.fits,
            "DATE-AVG": (start_time.tai + (stop_time.tai - start_time.tai) / 2.0).fits,
            "DATE-END": stop_time.tai.fits,
        }

    def get_hdu_header_base_observation_information(
        self, obs_id_only: bool = False
    ) -> Dict[str, Any]:
        """
        Return generic information on the observation setting (id, target, ...)

        Parameters
        ----------
        obs_id_only : bool
            If true, will return a dict with as only information the obs_id
        """
        if self.obs_id is None:
            raise ValueError("Observation ID is missing.")
        header = {"OBS_ID": self.obs_id}
        if self.target_information is not None and not obs_id_only:
            header["OBSERVER"] = self.target_information["observer"]
            header["OBJECT"] = self.target_information["object_name"]
            object_coordinate = self.target_information[
                "object_coordinate"
            ].transform_to(ICRS())
            if not np.isnan(object_coordinate.ra.to_value(u.deg)):
                header["RA_OBJ"] = object_coordinate.ra.to_value(u.deg)
            if not np.isnan(object_coordinate.dec.to_value(u.deg)):
                header["DEC_OBJ"] = object_coordinate.dec.to_value(u.deg)
        return header

    def get_hdu_header_base_subarray_information(self) -> Dict[str, Any]:
        """
        Return generic information on the array used for observations
        """
        if self.telescope_information is None:
            raise ValueError("Telescope information are missing.")
        header = {
            "ORIGIN": self.telescope_information["organisation"],
            "TELESCOP": self.telescope_information["array"],
            "INSTRUME": self.telescope_information["subarray"],
            "TELLIST": str(self.telescope_information["telescope_list"]),
            "N_TELS": len(self.telescope_information["telescope_list"]),
        }
        return header

    def get_hdu_header_base_software_information(self) -> Dict[str, Any]:
        """
        Return information about the software versions used to process the observation
        """
        header = {}
        if self.software_information is not None:
            header["DST_VER"] = self.software_information["dst_version"]
            header["ANA_VER"] = self.software_information["analysis_version"]
            header["CAL_VER"] = self.software_information["calibration_version"]
        return header

    def get_hdu_header_base_pointing(self) -> Dict[str, Any]:
        """
        Return information on the pointing during the observation
        """
        if self.pointing is None:
            raise ValueError("Pointing information are missing")
        if self.pointing_mode is None:
            raise ValueError("Pointing mode is missing")
        if self.location is None:
            raise ValueError("Telescope location information are missing")

        gti_table = self.create_gti_table()
        delta_time_evaluation = []
        for i in range(len(gti_table)):
            delta_time_evaluation += list(
                np.linspace(gti_table["START"][i], gti_table["STOP"][i], 100)
            )
        delta_time_evaluation = u.Quantity(delta_time_evaluation)
        time_evaluation = self._reference_time + TimeDelta(delta_time_evaluation)

        pointing_table = self.create_pointing_table()
        if self.pointing_mode == "TRACK":
            obs_mode = "POINTING"
            icrs_coordinate = SkyCoord(
                ra=np.interp(
                    delta_time_evaluation,
                    xp=pointing_table["TIME"],
                    fp=pointing_table["RA_PNT"],
                ),
                dec=np.interp(
                    delta_time_evaluation,
                    xp=pointing_table["TIME"],
                    fp=pointing_table["DEC_PNT"],
                ),
            )
            altaz_coordinate = icrs_coordinate.transform_to(
                AltAz(location=self.location, obstime=time_evaluation)
            )
        elif self.pointing_mode == "DRIFT":
            obs_mode = "DRIFT"
            altaz_coordinate = AltAz(
                alt=np.interp(
                    delta_time_evaluation,
                    xp=pointing_table["TIME"],
                    fp=pointing_table["ALT_PNT"],
                ),
                az=np.interp(
                    delta_time_evaluation,
                    xp=pointing_table["TIME"],
                    fp=pointing_table["AZ_PNT"],
                ),
                location=self.location,
                obstime=time_evaluation,
            )
            icrs_coordinate = altaz_coordinate.transform_to(ICRS())
        else:
            raise ValueError("Unknown pointing mode")

        header = {
            "RADECSYS": "ICRS",
            "EQUINOX": 2000.0,
            "OBS_MODE": obs_mode,
            "RA_PNT": np.mean(icrs_coordinate.ra.to_value(u.deg)),
            "DEC_PNT": np.mean(icrs_coordinate.dec.to_value(u.deg)),
            "ALT_PNT": np.mean(altaz_coordinate.alt.to_value(u.deg)),
            "AZ_PNT": np.mean(altaz_coordinate.az.to_value(u.deg)),
            "GEOLON": self.location.lon.to_value(u.deg),
            "GEOLAT": self.location.lat.to_value(u.deg),
            "ALTITUDE": self.location.height.to_value(u.m),
            "OBSGEO-X": self.location.x.to_value(u.m),
            "OBSGEO-Y": self.location.y.to_value(u.m),
            "OBSGEO-Z": self.location.z.to_value(u.m),
        }
        return header

    def get_hdu_header_events(self) -> Dict[str, Any]:
        """
        The output dictionary contain all the necessary information that should be added to the header of the events HDU
        """
        header = self.get_hdu_header_base_format()
        header.update({"HDUCLAS1": "EVENTS"})
        header.update(self.get_hdu_header_base_time())
        header.update(self.get_hdu_header_base_pointing())
        header.update(self.get_hdu_header_base_observation_information())
        header.update(self.get_hdu_header_base_subarray_information())
        header.update(self.get_hdu_header_base_software_information())
        return header

    def get_hdu_header_gti(self) -> Dict[str, Any]:
        """
        The output dictionary contain all the necessary information that should be added to the header of the GTI HDU
        """
        header = self.get_hdu_header_base_format()
        header.update({"HDUCLAS1": "GTI"})
        header.update(self.get_hdu_header_base_time())
        header.update(
            self.get_hdu_header_base_observation_information(obs_id_only=True)
        )
        return header

    def get_hdu_header_pointing(self) -> Dict[str, Any]:
        """
        The output dictionary contain all the necessary information that should be added to the header of the pointing HDU
        """
        header = self.get_hdu_header_base_format()
        header.update({"HDUCLAS1": "POINTING"})
        header.update(self.get_hdu_header_base_pointing())
        header.update(
            self.get_hdu_header_base_observation_information(obs_id_only=True)
        )
        return header

    def transform_events_columns_for_gadf_format(self, events: QTable) -> QTable:
        """
        Return an event table containing only the columns that should be added to the EVENTS HDU
        It also rename all the columns to match the name expected in the GADF format

        Parameters
        ----------
        events : QTable
            The base events table to process
        """
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
                "reco_x_max",
                "reco_x_max_uncert",
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
                "XMAX",
                "XMAX_ERR",
            ]

            for i, c in enumerate(rename_from_optional):
                if c not in events.colnames:
                    self.log.warning(
                        f"Optional column {c} is missing from the events table."
                    )
                    if self.raise_error_for_optional:
                        raise ValueError(
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
        """
        Build a table that contains GTI information with the GADF names and format, to be concerted directly as a TableHDU
        """
        table_structure = {"START": [], "STOP": []}
        for gti_interval in self.gti:
            table_structure["START"].append(
                (gti_interval[0].tai - self._reference_time).to(u.s)
            )
            table_structure["STOP"].append(
                (gti_interval[1].tai - self._reference_time).to(u.s)
            )

        QTable(table_structure).sort("START")
        for i in range(len(table_structure) - 1):
            if table_structure["STOP"][i] > table_structure["START"][i + 1]:
                self.log.warning("Overlapping GTI intervals")
                break

        return QTable(table_structure)

    def create_pointing_table(self) -> QTable:
        """
        Build a table that contains pointing information with the GADF names and format, to be concerted directly as a TableHDU
        """
        if self.pointing is None:
            raise ValueError("Pointing information are missing")
        if self.location is None:
            raise ValueError("Telescope location information are missing")

        table_structure = {
            "TIME": [],
            "RA_PNT": [],
            "DEC_PNT": [],
            "ALT_PNT": [],
            "AZ_PNT": [],
        }

        for pointing in self.pointing:
            time = pointing[0]
            pointing_icrs = pointing[1].transform_to(ICRS())
            pointing_altaz = pointing[1].transform_to(
                AltAz(location=self.location, obstime=time)
            )
            table_structure["TIME"].append((time.tai - self._reference_time).to(u.s))
            table_structure["RA_PNT"].append(pointing_icrs.ra.to(u.deg))
            table_structure["DEC_PNT"].append(pointing_icrs.dec.to(u.deg))
            table_structure["ALT_PNT"].append(pointing_altaz.alt.to(u.deg))
            table_structure["AZ_PNT"].append(pointing_altaz.az.to(u.deg))

        table = QTable(table_structure)
        table.sort("TIME")
        return table
