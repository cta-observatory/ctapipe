from astropy.io import fits

from ctapipe.core import Tool, traits
from ctapipe.core.traits import Bool, Integer, classes_with_traits, flag

from ..irf import EventLoader, EventPreprocessor
from ..irf.cuts import EventSelection
from ..irf.dl3 import DL3EventsWriter, DL3GADFEventsWriter

__all__ = ["DL3Tool"]


class DL3Tool(Tool):
    name = "ctapipe-create-dl3"
    description = "Create DL3 file from DL2 observation file"

    dl2_file = traits.Path(
        allow_none=False,
        directory_ok=False,
        exists=True,
        help="DL2 input filename and path.",
    ).tag(config=True)

    output_file = traits.Path(
        allow_none=False,
        directory_ok=False,
        help="Output file",
    ).tag(config=True)

    irfs_file = traits.Path(
        allow_none=False,
        directory_ok=False,
        exists=True,
        help="Path to the IRFs describing the observation",
    ).tag(config=True)

    chunk_size = Integer(
        default_value=100000,
        allow_none=True,
        help="How many subarray events to load at once while selecting.",
    ).tag(config=True)

    optional_dl3_columns = Bool(
        default_value=False, help="If true add optional columns to produce file"
    ).tag(config=True)

    raise_error_for_optional = Bool(
        default_value=True,
        help="If true will raise error in the case optional column are missing",
    ).tag(config=True)

    # Which classes are registered for configuration
    classes = (
        [
            EventLoader,
        ]
        + classes_with_traits(EventSelection)
        + classes_with_traits(EventPreprocessor)
    )

    aliases = {
        "cuts": "EventSelection.cuts_file",
        "dl2-file": "DL3Tool.dl2_file",
        "irfs-file": "DL3Tool.irfs_file",
        "output": "DL3Tool.output_file",
        "chunk-size": "DL3Tool.chunk_size",
    }

    flags = {
        **flag(
            "optional-columns",
            "DL3Tool.optional_dl3_columns",
            "Add optional columns for events in the DL3 file",
            "Do not add optional column for events in the DL3 file",
        ),
        **flag(
            "raise-error-for-optional",
            "DL3Tool.raise_error_for_optional",
            "Raise an error if an optional column is missing",
            "Only display a warning if an optional column is missing, it will lead to optional columns missing in the DL3 file",
        ),
    }

    def setup(self):
        """
        Initialize components from config and load g/h (and theta) cuts.
        """

        # Setting preprocessing for DL3
        EventPreprocessor.irf_pre_processing = False
        EventPreprocessor.optional_dl3_columns = self.optional_dl3_columns
        EventPreprocessor.raise_error_for_optional = self.raise_error_for_optional

        # Setting the GADF format object
        DL3EventsWriter.optional_dl3_columns = self.optional_dl3_columns
        DL3EventsWriter.raise_error_for_optional = self.raise_error_for_optional
        DL3EventsWriter.overwrite = self.overwrite

        self.dl3_format = DL3GADFEventsWriter()

    def start(self):
        self.log.info("Loading events from DL2")
        self.event_loader = EventLoader(
            parent=self, file=self.dl2_file, quality_selection_only=False
        )
        self.dl3_format.events = self.event_loader.load_preselected_events(
            self.chunk_size
        )
        meta = self.event_loader.get_observation_information()
        self.dl3_format.obs_id = meta["obs_id"]
        self.dl3_format.pointing = meta["pointing"]["pointing_list"]
        self.dl3_format.pointing_mode = meta["pointing"]["pointing_mode"]
        self.dl3_format.gti = meta["gti"]
        self.dl3_format.dead_time_fraction = meta["dead_time_fraction"]

        self.dl3_format.location = meta["location"]
        self.dl3_format.telescope_information = meta["telescope_information"]
        self.dl3_format.target_information = meta["target"]
        self.dl3_format.software_information = meta["software_version"]

        self.log.info("Loading IRFs")
        hdu_irfs = fits.open(self.irfs_file, checksum=True)
        for i in range(1, len(hdu_irfs)):
            if "HDUCLAS2" in hdu_irfs[i].header.keys():
                if hdu_irfs[i].header["HDUCLAS2"] == "EFF_AREA":
                    if self.dl3_format.aeff is None:
                        self.dl3_format.aeff = hdu_irfs[i]
                    elif "EXTNAME" in hdu_irfs[i].header and not (
                        "PROTONS" in hdu_irfs[i].header["EXTNAME"]
                        or "ELECTRONS" in hdu_irfs[i].header["EXTNAME"]
                    ):
                        self.dl3_format.aeff = hdu_irfs[i]
                elif hdu_irfs[i].header["HDUCLAS2"] == "EDISP":
                    self.dl3_format.edisp = hdu_irfs[i]
                elif hdu_irfs[i].header["HDUCLAS2"] == "PSF":
                    self.dl3_format.psf = hdu_irfs[i]
                elif hdu_irfs[i].header["HDUCLAS2"] == "BKG":
                    self.dl3_format.bkg = hdu_irfs[i]

        self.log.info("Writing DL3 File")
        self.dl3_format.write_file(self.output_file)

    def finish(self):
        pass


def main():
    tool = DL3Tool()
    tool.run()


if __name__ == "main":
    main()
