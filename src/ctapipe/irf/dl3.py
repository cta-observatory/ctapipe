from datetime import datetime

from astropy.io import fits
from astropy.io.fits.hdu.base import ExtensionHDU
from astropy.table import QTable

from ctapipe.compat import COPY_IF_NEEDED
from ctapipe.core import Component
from ctapipe.core.traits import Bool


class DL3_GADF(Component):
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

    overwrite = Bool(
        default_value=False,
        help="If true, allow to overwrite already existing output file",
    ).tag(config=False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._events = None
        self._pointing = None
        self._gti = None
        self._aeff = None
        self._psf = None
        self._edisp = None
        self._bkg = None

    def write_file(self, path):
        events = self.transform_events_columns_for_gadf_format(self.events)

        hdu_dl3 = fits.HDUList(
            [
                fits.PrimaryHDU(
                    header={"CREATED": datetime.now(tz=datetime.UTC).isoformat()}
                )
            ]
        )
        hdu_dl3.append(
            fits.BinTableHDU(
                data=events,
                name="EVENTS",
                header=self.get_hdu_header_events(),
            )
        )
        hdu_dl3.append(self.aeff)
        hdu_dl3.append(self.psf)
        hdu_dl3.append(self.edisp)
        hdu_dl3.append(self.bkg)

        hdu_dl3.writeto(path, checksum=True, overwrite=self.overwrite)

    @property
    def events(self):
        return self._events

    @events.setter
    def events(self, events: QTable):
        self._events = events

    @property
    def aeff(self):
        return self._aeff

    @aeff.setter
    def aeff(self, aeff: ExtensionHDU):
        self._aeff = aeff

    @property
    def psf(self):
        return self._psf

    @psf.setter
    def psf(self, psf: ExtensionHDU):
        self._psf = psf

    @property
    def bkg(self):
        return self._bkg

    @bkg.setter
    def bkg(self, bkg: ExtensionHDU):
        self._bkg = bkg

    def get_hdu_header_events(self):
        return {"HDUCLASS": "GADF", "HDUCLAS1": "EVENTS"}

    def transform_events_columns_for_gadf_format(self, events):
        rename_from = ["event_id", "time", "reco_ra", "reco_dec", "reco_energy"]
        rename_to = ["EVENT_ID", "TIME", "RA", "DEC", "ENERGY"]

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
