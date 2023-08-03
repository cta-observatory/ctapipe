from traitlets.config import Application

from .version import __version__


class MainTool(Application):
    """
    Main entry point for ctapipe, provides other tools as subcommands
    """

    name = "ctapipe"
    version = __version__

    subcommands = {
        "process": (
            "ctapipe.tools.process.ProcessorTool",
            "ctapipe event-wise data processing",
        ),
        "apply-models": (
            "ctapipe.tools.apply_models.ApplyModels",
            "Apply trained machine learning models",
        ),
        "train": (
            "ctapipe.tools.train.TrainTool",
            "train various reconstruction models",
        ),
        "merge": (
            "ctapipe.tools.merge.MergeTool",
            "Merge multiple ctapipe output files into one",
        ),
        "fileinfo": (
            "ctapipe.tools.fileinfo.FileInfoTool",
            "Obtain metadata and other information from ctapipe output files",
        ),
        "info": (
            "ctapipe.tools.info.InfoTool",
            "Print information about ctapipe and the current installation",
        ),
        "quickstart": (
            "ctapipe.tools.quickstart.QuickStartTool",
            "Create a directory with example configuration files",
        ),
        "dump-instrument": (
            "ctapipe.tools.dump_instrument.DumpInstrumentTool",
            "read the subarray description from any file with an EventSource\nand write "
            "it to a set of output files.",
        ),
    }

    def start(self):
        if self.subapp is None:
            if len(self.extra_args) > 0:
                print(f"Unknown sub-command {self.extra_args[0]}\n\n")
            self.print_subcommands()
            self.exit(1)

        self.subapp.run()

    def write_provenance(self):
        """This tool should not write any provenance log"""


def main():
    MainTool.launch_instance()


if __name__ == "__main__":
    main()
