from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from typing import Any, Dict, List, Tuple

import astropy.units as u
import numpy as np
from astropy.coordinates import (
    ICRS,
    AltAz,
    Angle,
    BaseCoordinateFrame,
    EarthLocation,
    SkyCoord,
)
from astropy.io import fits
from astropy.io.fits import Header
from astropy.io.fits.hdu.base import ExtensionHDU
from astropy.stats import circmean
from astropy.table import QTable
from astropy.time import Time, TimeDelta

from ..compat import COPY_IF_NEEDED
from ..core import Component
from ..core.traits import AstroTime, Bool
from ..version import version as ctapipe_version

__all__ = ["DL3EventsData", "DL3EventsWriter", "DL3GADFEventsWriter"]


@dataclass(slots=True)
class DL3EventsData:
    """
    The class contain all information required to generate DL3 file

    Parameters
    ----------
    events : QTable
        A table with a line for each event and column for each of the parameters required for the DL3 creation.
    obs_id : int
        Observation ID.
    pointing : list[tuple[Time, SkyCoord]]
        A list with for each entry containing the time at which the coordinate where evaluated and the associated coordinates.
    pointing_mode : str
        The name of the pointing mode used for the observation. Must be ``TRACK`` or ``DRIFT``.
    gti : list[tuple[Time, Time]]
        A list with for each entry containing the start and stop time of the good time intervals.
    livetime_fraction : float
        The livetime fraction for the observation.
    location : EarthLocation
        The location of the telescope.
    telescope_information : dict[str, any]
        A dictionary containing general information about telescope with as key: organisation, array, subarray, telescope_list.
    aeff : ExtensionHDU
        The effective area HDU read from the fits file containing IRFs.
    psf : ExtensionHDU
        The PSF HDU read from the fits file containing IRFs.
    edisp : ExtensionHDU
        The EDISP HDU read from the fits file containing IRFs.
    bkg : ExtensionHDU, optional
        The background HDU read from the fits file containing IRFs.
    target_information : dict[str, any], optional
        A dictionary containing general information about the targeted source with as key: observer, object_name, object_coordinate.
    software_information : dict[str, any], optional
        A dictionary containing general information about the software used to produce the file with as key: analysis_version, calibration_version, dst_version.
    """

    events: QTable
    obs_id: int
    pointing: List[Tuple[Time, SkyCoord]]
    pointing_mode: str
    gti: List[Tuple[Time, Time]]
    livetime_fraction: float
    location: EarthLocation
    telescope_information: Dict[str, Any]
    aeff: ExtensionHDU
    psf: ExtensionHDU
    edisp: ExtensionHDU
    bkg: ExtensionHDU | None = None
    target_information: Dict[str, Any] | None = None
    software_information: Dict[str, Any] | None = None

    def __setattr__(self, name: str, value: Any):
        """
        Set a DL3 payload field after validating its value.

        Parameters
        ----------
        name : str
            Name of the field to set.
        value : any
            New value for the field.
        """
        object.__setattr__(self, name, self._validate_field(name, value))

    def __post_init__(self):
        """
        Validate and normalize all DL3 payload fields after construction.
        """
        for field in fields(self):
            object.__setattr__(
                self,
                field.name,
                self._validate_field(field.name, getattr(self, field.name)),
            )

    def _validate_field(self, name: str, value: Any) -> Any:
        """
        Validate and normalize a DL3 payload field.

        Parameters
        ----------
        name : str
            Name of the field to validate.
        value : any
            Value to validate.

        Returns
        -------
        any
            The validated and normalized value.
        """
        validator = getattr(self, f"_validate_{name}", None)
        if validator is not None:
            value = validator(value)
        return value

    @staticmethod
    def _validate_obs_id(obs_id: int) -> int:
        """
        Validate observation ID.

        Parameters
        ----------
        obs_id : int
            Observation ID.

        Returns
        -------
        int
            Observation ID cast to a Python ``int``.
        """
        if obs_id is None:
            raise ValueError("obs_id is required.")
        if not isinstance(obs_id, (int, np.integer)) or isinstance(obs_id, bool):
            raise TypeError("obs_id must be an integer.")
        if obs_id < 0:
            raise ValueError("obs_id must be >= 0")
        return int(obs_id)

    @staticmethod
    def _validate_events(events: QTable) -> QTable:
        """
        Validate the events table.

        Parameters
        ----------
        events : QTable
            A table with a line for each event.

        Returns
        -------
        QTable
            The validated events table.
        """
        if events is None:
            raise ValueError("events is required.")
        if not isinstance(events, QTable):
            raise TypeError("events must be an astropy QTable.")
        return events

    @staticmethod
    def _validate_pointing(
        pointing: list[tuple[Time, SkyCoord]],
    ) -> list[tuple[Time, SkyCoord]]:
        """
        Validate the pointing information.

        Parameters
        ----------
        pointing : list[tuple[Time, SkyCoord]]
            A list with for each entry containing the time at which the
            coordinate where evaluated and the associated coordinates.

        Returns
        -------
        list[tuple[Time, SkyCoord]]
            The validated pointing information.
        """
        if pointing is None:
            raise ValueError("pointing is required.")
        if not isinstance(pointing, (list, tuple)):
            raise TypeError("pointing must be a list of (time, coordinate) pairs.")

        for i, value in enumerate(pointing):
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(f"pointing[{i}] must be a (time, coordinate) pair.")

            coordinate = value[1]
            if not isinstance(coordinate, (SkyCoord, BaseCoordinateFrame)):
                raise TypeError(
                    f"pointing[{i}].coordinate must be a SkyCoord or coordinate frame."
                )
        return pointing

    @staticmethod
    def _validate_pointing_mode(pointing_mode: str) -> str:
        """
        Validate and normalize the pointing mode.

        Parameters
        ----------
        pointing_mode : str
            The name of the pointing mode used for the observation.

        Returns
        -------
        str
            Pointing mode normalized to ``TRACK`` or ``DRIFT``.
        """
        if pointing_mode is None:
            raise ValueError("pointing_mode is required.")
        if not isinstance(pointing_mode, str):
            raise TypeError("pointing_mode must be a string.")

        pointing_mode = pointing_mode.strip().upper()
        if pointing_mode not in {"TRACK", "DRIFT"}:
            raise ValueError("pointing_mode must be either 'TRACK' or 'DRIFT'.")
        return pointing_mode

    @staticmethod
    def _validate_gti(gti: list[tuple[Time, Time]]) -> list[tuple[Time, Time]]:
        """
        Validate the good time intervals.

        Parameters
        ----------
        gti : list[tuple[Time, Time]]
            A list with for each entry containing the start and stop time of
            the good time intervals.

        Returns
        -------
        list[tuple[Time, Time]]
            The validated good time intervals.
        """
        if gti is None:
            raise ValueError("gti is required.")
        if not isinstance(gti, (list, tuple)):
            raise TypeError("gti must be a list of (start, stop) pairs.")

        for i, value in enumerate(gti):
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(f"gti[{i}] must be a (start, stop) pair.")
        return gti

    @staticmethod
    def _validate_livetime_fraction(livetime_fraction: float) -> float:
        """
        Validate the livetime fraction.

        Parameters
        ----------
        livetime_fraction : float
            The livetime fraction for the observations (DEADC correction
            factor).

        Returns
        -------
        float
            The validated livetime fraction.
        """
        if livetime_fraction is None:
            raise ValueError("livetime_fraction is required.")
        if isinstance(livetime_fraction, (bool, np.bool_)) or (
            not np.isscalar(livetime_fraction) or not np.isreal(livetime_fraction)
        ):
            raise TypeError("livetime_fraction must be a real scalar.")
        if not np.isfinite(livetime_fraction) or (not 0.0 <= livetime_fraction <= 1.0):
            raise ValueError("livetime_fraction must be in the range [0, 1].")
        return livetime_fraction

    @staticmethod
    def _validate_location(location: EarthLocation) -> EarthLocation:
        """
        Validate the telescope location.

        Parameters
        ----------
        location : EarthLocation
            The location of the telescope.

        Returns
        -------
        EarthLocation
            The validated telescope location.
        """
        if location is None:
            raise ValueError("location is required.")
        if not isinstance(location, EarthLocation):
            raise TypeError("location must be an astropy EarthLocation.")
        return location

    @staticmethod
    def _validate_telescope_information(
        telescope_information: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Validate the telescope information.

        Parameters
        ----------
        telescope_information : dict[str, any]
            A dictionary containing general information about telescope with as
            key: organisation, array, subarray, telescope_list.

        Returns
        -------
        dict[str, any]
            The validated telescope information.
        """
        if telescope_information is None:
            raise ValueError("telescope_information is required.")
        if not isinstance(telescope_information, Mapping):
            raise TypeError("telescope_information must be a mapping.")
        required = {"organisation", "array", "subarray", "telescope_list"}
        missing = required - set(telescope_information)
        if missing:
            raise ValueError(
                "telescope_information is missing keys: " + ", ".join(sorted(missing))
            )
        return telescope_information

    @staticmethod
    def _validate_target_information(
        target_information: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Validate the target information.

        Parameters
        ----------
        target_information : dict[str, any]
            A dictionary containing general information about the targeted
            source with as key: observer, object_name, object_coordinate.

        Returns
        -------
        dict[str, any] or None
            The validated target information, or ``None`` if omitted.
        """
        if target_information is None:
            return None
        if not isinstance(target_information, Mapping):
            raise TypeError("target_information must be a mapping.")
        required = {"observer", "object_name", "object_coordinate"}
        missing = required - set(target_information)
        if missing:
            raise ValueError(
                "target_information is missing keys: " + ", ".join(sorted(missing))
            )

        coordinate = target_information["object_coordinate"]
        if not isinstance(coordinate, SkyCoord):
            raise TypeError(
                "target_information['object_coordinate'] must be a SkyCoord"
            )
        return target_information

    @staticmethod
    def _validate_software_information(
        software_information: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Validate the software information.

        Parameters
        ----------
        software_information : dict[str, any]
            A dictionary containing general information about the software used
            to produce the file with as key: analysis_version,
            calibration_version, dst_version.

        Returns
        -------
        dict[str, any] or None
            The validated software information, or ``None`` if omitted.
        """
        if software_information is None:
            return None
        if not isinstance(software_information, Mapping):
            raise TypeError("software_information must be a mapping.")
        required = {"analysis_version", "calibration_version", "dst_version"}
        missing = required - set(software_information)
        if missing:
            raise ValueError(
                "software_information is missing keys: " + ", ".join(sorted(missing))
            )
        return software_information

    @staticmethod
    def _validate_irf(value: ExtensionHDU, name: str) -> ExtensionHDU:
        """
        Validate a required IRF HDU.

        Parameters
        ----------
        value : ExtensionHDU
            IRF HDU read from the fits file containing IRFs.
        name : str
            Name of the IRF field used in error messages.

        Returns
        -------
        ExtensionHDU
            The validated IRF HDU.
        """
        if value is None:
            raise ValueError(f"{name} is required.")
        if not isinstance(value, ExtensionHDU):
            raise TypeError(f"{name} must be a FITS ExtensionHDU.")
        return value

    def _validate_aeff(self, aeff: ExtensionHDU) -> ExtensionHDU:
        """
        Validate the effective area HDU.

        Parameters
        ----------
        aeff : ExtensionHDU
            The effective area HDU read from the fits file containing IRFs.

        Returns
        -------
        ExtensionHDU
            The validated effective area HDU.
        """
        return self._validate_irf(aeff, "aeff")

    def _validate_psf(self, psf: ExtensionHDU) -> ExtensionHDU:
        """
        Validate the PSF HDU.

        Parameters
        ----------
        psf : ExtensionHDU
            The PSF HDU read from the fits file containing IRFs.

        Returns
        -------
        ExtensionHDU
            The validated PSF HDU.
        """
        return self._validate_irf(psf, "psf")

    def _validate_edisp(self, edisp: ExtensionHDU) -> ExtensionHDU:
        """
        Validate the EDISP HDU.

        Parameters
        ----------
        edisp : ExtensionHDU
            The EDISP HDU read from the fits file containing IRFs.

        Returns
        -------
        ExtensionHDU
            The validated EDISP HDU.
        """
        return self._validate_irf(edisp, "edisp")

    @staticmethod
    def _validate_bkg(bkg: ExtensionHDU) -> ExtensionHDU:
        """
        Validate the background HDU.

        Parameters
        ----------
        bkg : ExtensionHDU
            The background HDU read from the fits file containing IRFs.

        Returns
        -------
        ExtensionHDU or None
            The validated background HDU, or ``None`` if omitted.
        """
        if bkg is not None and not isinstance(bkg, ExtensionHDU):
            raise TypeError("bkg must be a FITS ExtensionHDU.")
        return bkg


class DL3EventsWriter(Component):
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

    @abstractmethod
    def write_file(self, path: str, data: DL3EventsData):
        """
        This function will write the new DL3 file.

        Parameters
        ----------
        path : str
            The full path and filename of the new file to write.
        data : DL3EventsData
            The DL3 file payload to write.

        Returns
        -------
        None
        """
        pass


class DL3GADFEventsWriter(DL3EventsWriter):
    """
    Class to write DL3 in GADF format, subclass of DL3_Format
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._reference_time = self.reference_time.tai

    def write_file(self, path: str, data: DL3EventsData):
        """
        This function will write the new DL3 file.

        Parameters
        ----------
        path : str
            The full path and filename of the new file to write.
        data : DL3EventsData
            The DL3 file payload to write.

        Returns
        -------
        None
        """
        if not isinstance(data, DL3EventsData):
            raise TypeError("data must be a DL3EventsData instance.")

        creation_time = datetime.now(tz=UTC)
        hdu_dl3 = fits.HDUList(
            [
                fits.PrimaryHDU(
                    header=Header(self.get_hdu_header_base_format(creation_time))
                )
            ]
        )
        hdu_dl3.append(
            fits.BinTableHDU(
                data=self.transform_events_columns_for_gadf_format(data.events),
                name="EVENTS",
                header=Header(self.get_hdu_header_events(data, creation_time)),
            )
        )
        hdu_dl3.append(
            fits.BinTableHDU(
                data=self.create_gti_table(data),
                name="GTI",
                header=Header(self.get_hdu_header_gti(data, creation_time)),
            )
        )
        hdu_dl3.append(
            fits.BinTableHDU(
                data=self.create_pointing_table(data),
                name="POINTING",
                header=Header(self.get_hdu_header_pointing(data, creation_time)),
            )
        )

        if data.aeff is None:
            raise ValueError("Missing effective area IRF")
        if data.psf is None:
            raise ValueError("Missing PSF IRF")
        if data.edisp is None:
            raise ValueError("Missing EDISP IRF")

        for irf in (data.aeff, data.psf, data.edisp):
            output_hdu = irf.copy()
            output_hdu.header["OBS_ID"] = data.obs_id
            hdu_dl3.append(output_hdu)

        if data.bkg is not None:
            output_hdu = data.bkg.copy()
            output_hdu.header["OBS_ID"] = data.obs_id
            hdu_dl3.append(output_hdu)

        hdu_dl3.writeto(path, checksum=True, overwrite=self.overwrite)

    def get_hdu_header_base_format(
        self, creation_time: datetime | None = None
    ) -> Dict[str, Any]:
        """
        Return the base information that should be included in all HDU of the final fits file.

        Parameters
        ----------
        creation_time : datetime, optional
            The file creation time to write into the header. If omitted, the
            current UTC time is used.

        Returns
        -------
        dict[str, any]
            Header keywords common to all HDUs in the DL3 file.
        """
        return {
            "HDUCLASS": "GADF",
            "HDUVERS": "v0.3",
            "HDUDOC": "https://gamma-astro-data-formats.readthedocs.io/en/v0.3/index.html",
            "CREATOR": "ctapipe " + ctapipe_version,
            "CREATED": (creation_time or datetime.now(tz=UTC)).isoformat(),
        }

    def get_hdu_header_time_reference(self) -> Dict[str, Any]:
        """
        Return the time reference keywords needed to interpret TIME columns.

        These keywords (MJDREFI, MJDREFF, TIMEUNIT, TIMESYS, TIMEREF) should
        be present in every HDU that contains a TIME column or time-related
        header values.

        Returns
        -------
        dict[str, any]
            Header keywords defining the FITS time reference.
        """
        return {
            "MJDREFI": int(self._reference_time.mjd),
            "MJDREFF": self._reference_time.mjd % 1,
            "TIMEUNIT": "s",
            "TIMEREF": "TOPOCENTER",
            "TIMESYS": "TAI",
        }

    def get_hdu_header_base_time(self, data: DL3EventsData) -> Dict[str, Any]:
        """
        Return the information about time parameters used in several HDU.

        Parameters
        ----------
        data : DL3EventsData
            The DL3 file payload containing the GTI and livetime fraction.

        Returns
        -------
        dict[str, any]
            Header keywords describing the observation time range and livetime.
        """
        if data.gti is None:
            raise ValueError("No available time information for the DL3 file")
        if data.livetime_fraction is None:
            raise ValueError("No available livetime fraction for the DL3 file")

        start_time = None
        stop_time = None
        ontime = TimeDelta(0.0 * u.s)
        for i, gti_interval in enumerate(data.gti):
            interval_start = self._to_tai_time(gti_interval[0], f"gti[{i}].start")
            interval_stop = self._to_tai_time(gti_interval[1], f"gti[{i}].stop")
            if interval_stop < interval_start:
                raise ValueError(
                    f"Invalid GTI interval at index {i}: stop time is before start time."
                )

            ontime += interval_stop - interval_start
            start_time = (
                interval_start
                if start_time is None
                else min(start_time, interval_start)
            )
            stop_time = (
                interval_stop if stop_time is None else max(stop_time, interval_stop)
            )

        header = self.get_hdu_header_time_reference()
        header.update(
            {
                "TSTART": self._to_relative_time_seconds(
                    start_time, "observation start"
                ),
                "TSTOP": self._to_relative_time_seconds(stop_time, "observation stop"),
                "ONTIME": ontime.to_value(u.s),
                "LIVETIME": ontime.to_value(u.s) * data.livetime_fraction,
                "DEADC": data.livetime_fraction,
                "TELAPSE": (stop_time - start_time).to_value(u.s),
                "DATE-OBS": start_time.fits,
                "DATE-BEG": start_time.fits,
                "DATE-AVG": (start_time + (stop_time - start_time) / 2.0).fits,
                "DATE-END": stop_time.fits,
            }
        )
        return header

    def get_hdu_header_base_observation_information(
        self, data: DL3EventsData, obs_id_only: bool = False
    ) -> Dict[str, Any]:
        """
        Return generic information on the observation setting (id, target, ...).

        Parameters
        ----------
        data : DL3EventsData
            The DL3 file payload containing the observation and target
            information.
        obs_id_only : bool
            If true, will return a dict with as only information the obs_id.

        Returns
        -------
        dict[str, any]
            Header keywords describing the observation and, if requested, the
            target information.
        """
        if data.obs_id is None:
            raise ValueError("Observation ID is missing.")
        header = {"OBS_ID": data.obs_id}
        if data.target_information is not None and not obs_id_only:
            header["OBSERVER"] = data.target_information["observer"]
            header["OBJECT"] = data.target_information["object_name"]
            object_coordinate = data.target_information[
                "object_coordinate"
            ].transform_to(ICRS())
            if not np.isnan(object_coordinate.ra.to_value(u.deg)):
                header["RA_OBJ"] = object_coordinate.ra.to_value(u.deg)
            if not np.isnan(object_coordinate.dec.to_value(u.deg)):
                header["DEC_OBJ"] = object_coordinate.dec.to_value(u.deg)
        return header

    def get_hdu_header_base_subarray_information(
        self, data: DL3EventsData
    ) -> Dict[str, Any]:
        """
        Return generic information on the array used for observations.

        Parameters
        ----------
        data : DL3EventsData
            The DL3 file payload containing the telescope information.

        Returns
        -------
        dict[str, any]
            Header keywords describing the array and telescope list.
        """
        if data.telescope_information is None:
            raise ValueError("Telescope information are missing.")
        header = {
            "ORIGIN": data.telescope_information["organisation"],
            "TELESCOP": data.telescope_information["array"],
            "INSTRUME": data.telescope_information["subarray"],
            "TELLIST": str(data.telescope_information["telescope_list"]),
            "N_TELS": len(data.telescope_information["telescope_list"]),
        }
        return header

    def get_hdu_header_base_software_information(
        self, data: DL3EventsData
    ) -> Dict[str, Any]:
        """
        Return information about the software versions used to process the observation.

        Parameters
        ----------
        data : DL3EventsData
            The DL3 file payload containing the software information.

        Returns
        -------
        dict[str, any]
            Header keywords describing software versions used to process the
            observation.
        """
        header = {}
        if data.software_information is not None:
            header["DST_VER"] = data.software_information["dst_version"]
            header["ANA_VER"] = data.software_information["analysis_version"]
            header["CAL_VER"] = data.software_information["calibration_version"]
        return header

    def get_hdu_header_base_pointing(self, data: DL3EventsData) -> Dict[str, Any]:
        """
        Return information on the pointing during the observation.

        Parameters
        ----------
        data : DL3EventsData
            The DL3 file payload containing pointing, pointing mode, GTI and
            telescope location information.

        Returns
        -------
        dict[str, any]
            Header keywords describing the observation pointing and telescope
            location.
        """
        if data.pointing is None:
            raise ValueError("Pointing information are missing")
        if data.pointing_mode is None:
            raise ValueError("Pointing mode is missing")
        if data.location is None:
            raise ValueError("Telescope location information are missing")

        gti_table = self.create_gti_table(data)
        delta_time_evaluation = []
        for i in range(len(gti_table)):
            delta_time_evaluation += list(
                np.linspace(gti_table["START"][i], gti_table["STOP"][i], 100)
            )
        delta_time_evaluation = u.Quantity(delta_time_evaluation)
        time_evaluation = self._reference_time + TimeDelta(delta_time_evaluation)

        pointing_table = self.create_pointing_table(data)
        if data.pointing_mode == "TRACK":
            obs_mode = "POINTING"
            icrs_coordinate = SkyCoord(
                ra=self._circular_interp(
                    delta_time_evaluation,
                    xp=pointing_table["TIME"],
                    fp_deg=pointing_table["RA_PNT"],
                ),
                dec=np.interp(
                    delta_time_evaluation,
                    xp=pointing_table["TIME"],
                    fp=pointing_table["DEC_PNT"],
                ),
                unit=u.deg,
            )
            altaz_coordinate = icrs_coordinate.transform_to(
                AltAz(location=data.location, obstime=time_evaluation)
            )
        elif data.pointing_mode == "DRIFT":
            obs_mode = "DRIFT"
            altaz_coordinate = AltAz(
                alt=u.Quantity(
                    np.interp(
                        delta_time_evaluation,
                        xp=pointing_table["TIME"],
                        fp=pointing_table["ALT_PNT"],
                    ),
                    u.deg,
                    copy=COPY_IF_NEEDED,
                ),
                az=self._circular_interp(
                    delta_time_evaluation,
                    xp=pointing_table["TIME"],
                    fp_deg=pointing_table["AZ_PNT"],
                )
                * u.deg,
                location=data.location,
                obstime=time_evaluation,
            )
            icrs_coordinate = altaz_coordinate.transform_to(ICRS())
        else:
            raise ValueError("Unknown pointing mode")

        header = {
            "RADESYS": "ICRS",
            "RADECSYS": "ICRS",
            "EQUINOX": 2000.0,
            "OBS_MODE": obs_mode,
            "RA_PNT": Angle(circmean(icrs_coordinate.ra))
            .wrap_at(360 * u.deg)
            .to_value(u.deg),
            "DEC_PNT": np.mean(icrs_coordinate.dec.to_value(u.deg)),
            "ALT_PNT": np.mean(altaz_coordinate.alt.to_value(u.deg)),
            "AZ_PNT": Angle(circmean(altaz_coordinate.az))
            .wrap_at(360 * u.deg)
            .to_value(u.deg),
            "GEOLON": data.location.lon.to_value(u.deg),
            "GEOLAT": data.location.lat.to_value(u.deg),
            "ALTITUDE": data.location.height.to_value(u.m),
            "OBSGEO-X": data.location.x.to_value(u.m),
            "OBSGEO-Y": data.location.y.to_value(u.m),
            "OBSGEO-Z": data.location.z.to_value(u.m),
        }
        return header

    def get_hdu_header_events(
        self, data: DL3EventsData, creation_time=None
    ) -> Dict[str, Any]:
        """
        Return all the necessary information that should be added to the header of the events HDU.

        Parameters
        ----------
        data : DL3EventsData
            The DL3 file payload to use for the header.
        creation_time : datetime, optional
            The file creation time to write into the header. If omitted, the
            current UTC time is used.

        Returns
        -------
        dict[str, any]
            Header keywords for the EVENTS HDU.
        """
        header = self.get_hdu_header_base_format(creation_time)
        header.update({"HDUCLAS1": "EVENTS", "FOVALIGN": "ALTAZ"})
        header.update(self.get_hdu_header_base_time(data))
        header.update(self.get_hdu_header_base_pointing(data))
        header.update(self.get_hdu_header_base_observation_information(data))
        header.update(self.get_hdu_header_base_subarray_information(data))
        header.update(self.get_hdu_header_base_software_information(data))
        return header

    def get_hdu_header_gti(
        self, data: DL3EventsData, creation_time=None
    ) -> Dict[str, Any]:
        """
        Return all the necessary information that should be added to the header of the GTI HDU.

        Parameters
        ----------
        data : DL3EventsData
            The DL3 file payload to use for the header.
        creation_time : datetime, optional
            The file creation time to write into the header. If omitted, the
            current UTC time is used.

        Returns
        -------
        dict[str, any]
            Header keywords for the GTI HDU.
        """
        header = self.get_hdu_header_base_format(creation_time)
        header.update({"HDUCLAS1": "GTI"})
        header.update(self.get_hdu_header_base_time(data))
        header.update(
            self.get_hdu_header_base_observation_information(data, obs_id_only=True)
        )
        return header

    def get_hdu_header_pointing(
        self, data: DL3EventsData, creation_time=None
    ) -> Dict[str, Any]:
        """
        Return all the necessary information that should be added to the header of the pointing HDU.

        Parameters
        ----------
        data : DL3EventsData
            The DL3 file payload to use for the header.
        creation_time : datetime, optional
            The file creation time to write into the header. If omitted, the
            current UTC time is used.

        Returns
        -------
        dict[str, any]
            Header keywords for the POINTING HDU.
        """
        header = self.get_hdu_header_base_format(creation_time)
        header.update({"HDUCLAS1": "POINTING"})
        header.update(self.get_hdu_header_time_reference())
        header.update(self.get_hdu_header_base_pointing(data))
        header.update(
            self.get_hdu_header_base_observation_information(data, obs_id_only=True)
        )
        return header

    def transform_events_columns_for_gadf_format(self, events: QTable) -> QTable:
        """
        Return an event table containing only the columns that should be added to the EVENTS HDU
        It also rename all the columns to match the name expected in the GADF format

        Parameters
        ----------
        events : QTable
            The base events table to process.

        Returns
        -------
        QTable
            Event table containing the DL3/GADF columns with GADF names.
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
                "GAMMANESS",
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
        renamed_events["time"] = self._to_relative_time_quantity(
            renamed_events["time"], "events.time"
        )
        renamed_events.rename_columns(rename_from, rename_to)
        renamed_events = renamed_events[rename_to]
        return renamed_events

    def create_gti_table(self, data: DL3EventsData) -> QTable:
        """
        Build a table that contains GTI information with the GADF names and format, to be concerted directly as a TableHDU.

        Parameters
        ----------
        data : DL3EventsData
            The DL3 file payload containing the good time intervals.

        Returns
        -------
        QTable
            GTI table with START and STOP columns in seconds relative to the
            writer reference time.
        """
        table_structure = {"START": [], "STOP": []}
        for i, gti_interval in enumerate(data.gti):
            interval_start = self._to_tai_time(gti_interval[0], f"gti[{i}].start")
            interval_stop = self._to_tai_time(gti_interval[1], f"gti[{i}].stop")
            table_structure["START"].append(
                self._to_relative_time_quantity(interval_start, f"gti[{i}].start")
            )
            table_structure["STOP"].append(
                self._to_relative_time_quantity(interval_stop, f"gti[{i}].stop")
            )

        table = QTable(table_structure)
        table.sort("START")
        for i in range(len(table) - 1):
            if table["STOP"][i] > table["START"][i + 1]:
                self.log.warning("Overlapping GTI intervals")
                break

        return table

    def create_pointing_table(self, data: DL3EventsData) -> QTable:
        """
        Build a table that contains pointing information with the GADF names and format, to be concerted directly as a TableHDU.

        Parameters
        ----------
        data : DL3EventsData
            The DL3 file payload containing pointing and telescope location
            information.

        Returns
        -------
        QTable
            Pointing table with TIME, RA_PNT, DEC_PNT, ALT_PNT and AZ_PNT
            columns in GADF format.
        """
        if data.pointing is None:
            raise ValueError("Pointing information are missing")
        if data.location is None:
            raise ValueError("Telescope location information are missing")

        table_structure = {
            "TIME": [],
            "RA_PNT": [],
            "DEC_PNT": [],
            "ALT_PNT": [],
            "AZ_PNT": [],
        }

        for i, pointing in enumerate(data.pointing):
            time = self._to_tai_time(pointing[0], f"pointing[{i}].time")
            pointing_icrs = pointing[1].transform_to(ICRS())
            pointing_altaz = pointing[1].transform_to(
                AltAz(location=data.location, obstime=time)
            )
            table_structure["TIME"].append(
                self._to_relative_time_quantity(time, f"pointing[{i}].time")
            )
            table_structure["RA_PNT"].append(pointing_icrs.ra.to(u.deg))
            table_structure["DEC_PNT"].append(pointing_icrs.dec.to(u.deg))
            table_structure["ALT_PNT"].append(pointing_altaz.alt.to(u.deg))
            table_structure["AZ_PNT"].append(pointing_altaz.az.to(u.deg))

        table = QTable(table_structure)
        table.sort("TIME")
        return table

    def _to_tai_time(self, value: Any, value_name: str) -> Time:
        """
        Normalize input to an absolute TAI ``Time`` object.

        Parameters
        ----------
        value : Any
            Input time-like value. Supported types are ``Time``, ``TimeDelta``,
            time ``Quantity`` and scalar numeric values interpreted as seconds
            relative to ``reference_time``.
        value_name : str
            Name of the value used in error messages.

        Returns
        -------
        Time
            Input value converted to an absolute TAI time.
        """
        if isinstance(value, Time):
            return value.tai

        if isinstance(value, TimeDelta):
            return self._reference_time + value

        if isinstance(value, u.Quantity):
            if not value.unit.is_equivalent(u.s):
                raise ValueError(
                    f"{value_name} must be a time quantity equivalent to seconds."
                )
            return self._reference_time + TimeDelta(value.to(u.s))

        if np.isscalar(value) and np.isreal(value):
            return self._reference_time + TimeDelta(float(value) * u.s)

        raise TypeError(
            f"{value_name} must be Time, TimeDelta, a time Quantity, or a scalar number of seconds."
        )

    def _to_relative_time_seconds(self, value: Any, value_name: str) -> Any:
        """
        Normalize input to seconds relative to ``reference_time``.

        Parameters
        ----------
        value : Any
            Input time-like value. Supported types are ``Time``, ``TimeDelta``,
            time ``Quantity`` and numeric values assumed to already be in seconds.
        value_name : str
            Name of the value used in error messages.

        Returns
        -------
        float or numpy.ndarray
            Input value converted to seconds relative to the writer reference
            time.
        """
        if isinstance(value, Time):
            return (value.tai - self._reference_time).to_value(u.s)

        if isinstance(value, TimeDelta):
            return value.to_value(u.s)

        if isinstance(value, u.Quantity):
            if not value.unit.is_equivalent(u.s):
                raise ValueError(
                    f"{value_name} must be a time quantity equivalent to seconds."
                )
            return value.to_value(u.s)

        values = np.asarray(value)
        if np.issubdtype(values.dtype, np.number):
            return values.astype(np.float64, copy=False)

        raise TypeError(
            f"{value_name} must be Time, TimeDelta, a time Quantity, or numeric seconds."
        )

    def _to_relative_time_quantity(self, value: Any, value_name: str) -> u.Quantity:
        """
        Normalize input to a quantity in seconds relative to ``reference_time``.

        Parameters
        ----------
        value : Any
            Input time-like value. Supported types are ``Time``, ``TimeDelta``,
            time ``Quantity`` and numeric values assumed to already be in seconds.
        value_name : str
            Name of the value used in error messages.

        Returns
        -------
        astropy.units.Quantity
            Input value converted to seconds relative to the writer reference
            time.
        """
        return u.Quantity(
            self._to_relative_time_seconds(value, value_name),
            u.s,
            copy=COPY_IF_NEEDED,
        )

    @staticmethod
    def _circular_interp(x, xp, fp_deg):
        """
        Interpolate angular values in degrees, handling the 0/360 wrap-around.

        Uses ``np.unwrap`` to remove discontinuities before interpolation,
        then wraps the result back to [0, 360).

        Parameters
        ----------
        x : array-like
            The x-coordinates at which to evaluate the interpolated values.
        xp : array-like
            The x-coordinates of the data points (must be increasing).
        fp_deg : array-like
            The y-coordinates of the data points, in degrees.

        Returns
        -------
        np.ndarray
            Interpolated angular values in degrees, in [0, 360).
        """
        fp_rad = np.deg2rad(np.asarray(fp_deg, dtype=float))
        fp_unwrapped = np.unwrap(fp_rad)
        result_rad = np.interp(x, xp, fp_unwrapped)
        return np.rad2deg(result_rad) % 360
