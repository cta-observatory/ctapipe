from ctapipe.core import Tool, traits
from ctapipe.core.traits import Integer, classes_with_traits

from ..irf import EventLoader

__all__ = ["DL3Tool"]

from ..irf.cuts import EventSelection


class DL3Tool(Tool):
    name = "ctapipe-create-dl3"
    description = "Create DL3 file from DL2 observation file"

    dl2_file = traits.Path(
        allow_none=False,
        directory_ok=False,
        exists=True,
        help="DL2 input filename and path.",
    ).tag(config=True)

    output_path = traits.Path(
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

    # Which classes are registered for configuration
    classes = [
        EventLoader,
    ] + classes_with_traits(EventSelection)

    aliases = {
        "cuts": "EventSelection.cuts_file",
        "dl2-file": "DL3Tool.dl2_file",
        "irfs-file": "DL3Tool.irfs_file",
        "output": "DL3Tool.output_path",
        "chunk-size": "DL3Tool.chunk_size",
    }

    def setup(self):
        """
        Initialize components from config and load g/h (and theta) cuts.
        """
        self.log.info("Loading events from DL2")
        self.event_loader = EventLoader(
            parent=self, file=self.dl2_file, quality_selection_only=True
        )
        print(self.event_loader.load_preselected_events(self.chunk_size))

    def start(self):
        pass

    def finish(self):
        self.log.warning("Shutting down.")


def main():
    tool = DL3Tool()
    tool.run()


if __name__ == "main":
    main()
