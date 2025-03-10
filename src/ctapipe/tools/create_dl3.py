from astropy.io import fits

from ctapipe.core import Tool, traits
from ctapipe.core.traits import Bool, Integer, classes_with_traits, flag

from ..irf import EventLoader, EventPreprocessor
from ..irf.cuts import EventSelection
from ..irf.dl3 import DL3_GADF, DL3_Format

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
            "overwrite",
            "DL3_GADF.overwrite",
            "Will allow to overwrite existing DL3 file",
            "Prevent overwriting existing DL3 file",
        ),
        **flag(
            "optional-column",
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
        DL3_Format.optional_dl3_columns = self.optional_dl3_columns
        DL3_Format.raise_error_for_optional = self.raise_error_for_optional

        self.dl3_format = DL3_GADF()

    def start(self):
        self.log.info("Loading events from DL2")
        self.event_loader = EventLoader(
            parent=self, file=self.dl2_file, quality_selection_only=False
        )
        self.dl3_format.events = self.event_loader.load_preselected_events(
            self.chunk_size
        )
        array_location, gti = self.event_loader.get_observation_information()
        self.dl3_format.location = array_location
        self.dl3_format.gti = gti

        self.log.info("Loading IRFs")
        hdu_irfs = fits.open(self.irfs_file, checksum=True)
        for i in range(1, len(hdu_irfs)):
            if "HDUCLAS2" in hdu_irfs[i].header.keys():
                if hdu_irfs[i].header["HDUCLAS2"] == "EFF_AREA":
                    self.dl3_format.aeff = hdu_irfs[i]
                elif hdu_irfs[i].header["HDUCLAS2"] == "EDISP":
                    self.dl3_format.edisp = hdu_irfs[i]
                elif hdu_irfs[i].header["HDUCLAS2"] == "RPSF":
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
