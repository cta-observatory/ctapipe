from ctapipe.core import Tool, traits
from ctapipe.core.traits import (
    Integer,
)

from ..irf import (
    EventLoader,
    OptimizationResult,
)

__all__ = ["DL3Tool"]


class DL3Tool(Tool):
    name = "mytool"
    description = "do some things and stuff"
    aliases = dict(
        infile="AdvancedComponent.infile",
        outfile="AdvancedComponent.outfile",
        iterations="MyTool.iterations",
    )

    dl2_file = traits.Path(
        default_value=None,
        allow_none=False,
        directory_ok=False,
        exists=True,
        help="DL2 input filename and path.",
    ).tag(config=True)

    output_path = traits.Path(
        default_value=None,
        allow_none=False,
        directory_ok=False,
        exists=True,
        help="Output file",
    ).tag(config=True)

    cuts_file = traits.Path(
        default_value=None,
        allow_none=False,
        directory_ok=False,
        exists=True,
        help="Path to the cuts file to apply to the observation.",
    ).tag(config=True)

    irfs_file = traits.Path(
        default_value=None,
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
    ]

    def setup(self):
        """
        Initialize components from config and load g/h (and theta) cuts.
        """
        self.log.info("Loading cuts")
        self.opt_result = OptimizationResult.read(self.cuts_file)
        self.log.info("Loading events from DL2")
        self.event_loader = EventLoader(
            parent=self,
            file=self.dl2_file,
        )

    def start(self):
        pass

    def finish(self):
        self.log.warning("Shutting down.")


def main():
    tool = DL3Tool()
    tool.run()


if __name__ == "main":
    main()
