from datetime import datetime

from astropy.io import fits

from ctapipe.core import Tool, traits
from ctapipe.core.traits import Bool, Integer, classes_with_traits, flag

from ..irf import EventLoader, EventPreprocessor

__all__ = ["DL3Tool"]

from ..irf.cuts import EventSelection
from ..irf.dl3 import get_hdu_header_events


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

    overwrite = Bool(
        default_value=False,
        help="If true, allow to overwrite already existing output file",
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
        "overwrite": "DL3Tool.overwrite",
    }

    flags = {
        **flag(
            "optional-column",
            "EventPreprocessor.optional_dl3_columns",
            "Add optional columns for events in the DL3 file",
            "Do not add optional column for events in the DL3 file",
        ),
        **flag(
            "raise-error-for-optional",
            "EventPreprocessor.raise_error_for_optional",
            "Raise an error if an optional column is missing",
            "Only display a warning if an optional column is missing, it will lead to optional columns missing in the DL3 file",
        ),
    }

    def setup(self):
        """
        Initialize components from config and load g/h (and theta) cuts.
        """

        # Force the preprocessing for DL3
        EventPreprocessor.irf_pre_processing = False

    def start(self):
        self.log.info("Loading events from DL2")
        self.event_loader = EventLoader(
            parent=self, file=self.dl2_file, quality_selection_only=False
        )
        events = self.event_loader.load_preselected_events(self.chunk_size)

        hdu_dl3 = fits.HDUList(
            [
                fits.PrimaryHDU(
                    header={"CREATED": datetime.now(tz=datetime.UTC).isoformat()}
                )
            ]
        )
        hdu_dl3.append(
            fits.BinTableHDU(data=events, name="EVENTS", header=get_hdu_header_events())
        )

        self.log.info("Loading IRFs")
        hdu_irfs = fits.open(self.irfs_file, checksum=True)
        for i in range(1, len(hdu_irfs)):
            hdu_dl3.append(hdu_irfs[i])

        self.log.info("Writing DL3 File")
        hdu_dl3.writeto(self.output_file, checksum=True, overwrite=self.overwrite)

    def finish(self):
        pass


def main():
    tool = DL3Tool()
    tool.run()


if __name__ == "main":
    main()
