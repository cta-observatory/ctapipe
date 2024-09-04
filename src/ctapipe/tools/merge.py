"""
Merge multiple ctapipe HDF5 files into one
"""
import sys
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

from tqdm.auto import tqdm
from traitlets import List

from ctapipe.core.tool import ToolConfigurationError
from ctapipe.io.hdf5merger import CannotMerge

from ..core import Provenance, Tool, traits
from ..core.traits import Bool, Unicode, flag
from ..io import HDF5Merger
from ..io import metadata as meta


class MergeTool(Tool):
    """
    Merge multiple ctapipe HDF5 files into one
    """

    name = "ctapipe-merge"
    description = "Merge multiple ctapipe HDF5 files into one"

    examples = """
    To merge several files in the current directory:

    > ctapipe-merge file1.h5 file2.h5 file3.h5 --output=/path/output_file.h5 --progress

    For merging files from a specific directory with a specific pattern, use:

    > ctapipe-merge --input-dir=/input/dir/ --output=/path/output_file.h5 --progress
    --pattern='*.dl1.h5'

    If no pattern is given, all .h5 files in the given directory will be taken as input.
    """
    input_dir = traits.Path(
        default_value=None,
        help="Input directory",
        allow_none=True,
        exists=True,
        directory_ok=True,
        file_ok=False,
    ).tag(config=True)

    input_files = List(
        traits.Path(exists=True, directory_ok=False),
        default_value=[],
        help="Input CTA HDF5 files",
    ).tag(config=True)

    progress_bar = Bool(
        False,
        help="Show progress bar during processing",
    ).tag(config=True)

    file_pattern = Unicode(
        default_value="*.h5",
        help="Give a specific file pattern for matching files in ``input_dir``",
    ).tag(config=True)

    skip_broken_files = Bool(
        False,
        help="Skip files that cannot be merged instead of raising an error",
    ).tag(config=True)

    parser = ArgumentParser()
    parser.add_argument("input_files", nargs="*", type=Path)

    aliases = {
        ("i", "input-dir"): "MergeTool.input_dir",
        ("o", "output"): "HDF5Merger.output_path",
        ("p", "pattern"): "MergeTool.file_pattern",
    }

    flags = {
        "progress": (
            {"MergeTool": {"progress_bar": True}},
            "Show a progress bar for all given input files",
        ),
        "overwrite": (
            {"HDF5Merger": {"overwrite": True}},
            "Overwrite existing files",
        ),
        "append": (
            {"HDF5Merger": {"append": True}},
            "Append to existing files",
        ),
        **flag(
            "telescope-events",
            "HDF5Merger.telescope_events",
            "Include telescope-wise data",
            "Exclude telescope-wise data",
        ),
        **flag(
            "r0-waveforms",
            "HDF5Merger.r0_waveforms",
            "Include r0 waveforms",
            "Exclude r0 waveforms",
        ),
        **flag(
            "r1-waveforms",
            "HDF5Merger.r1_waveforms",
            "Include r1 waveforms",
            "Exclude r1 waveforms",
        ),
        **flag(
            "dl1-images",
            "HDF5Merger.dl1_images",
            "Include dl1 images",
            "Exclude dl1 images",
        ),
        **flag(
            "true-images",
            "HDF5Merger.true_images",
            "Include true images",
            "Exclude true images",
        ),
        **flag(
            "true-parameters",
            "HDF5Merger.true_parameters",
            "Include true parameters",
            "Exclude true parameters",
        ),
        **flag(
            "dl1-parameters",
            "HDF5Merger.dl1_parameters",
            "Include dl1 parameters",
            "Exclude dl1 parameters",
        ),
        **flag(
            "dl2-telescope",
            "HDF5Merger.dl2_telescope",
            "Include dl2 telescope-wise data",
            "Exclude dl2 telescope-wise data",
        ),
        **flag(
            "dl2-subarray",
            "HDF5Merger.dl2_subarray",
            "Include dl2 subarray-wise data",
            "Exclude dl2 subarray-wise data",
        ),
    }

    classes = [HDF5Merger]

    def setup(self):
        # Get input Files
        args = self.parser.parse_args(self.extra_args)
        self.input_files.extend(args.input_files)
        if self.input_dir is not None:
            self.input_files.extend(sorted(self.input_dir.glob(self.file_pattern)))

        if not self.input_files:
            self.log.critical(
                "No input files provided, either provide --input-dir "
                "or input files as positional arguments"
            )
            sys.exit(1)

        counts = Counter(self.input_files)
        duplicated = [p for p, c in counts.items() if c > 1]
        if len(duplicated) > 0:
            raise ToolConfigurationError(
                f"Same file given multiple times. Duplicated files are: {duplicated}"
            )

        self.merger = self.enter_context(HDF5Merger(parent=self))
        if self.merger.output_path in self.input_files:
            raise ToolConfigurationError(
                "Output path contained in input files. Fix your configuration / cli arguments."
            )

    def start(self):
        n_merged = 0

        for input_path in tqdm(
            self.input_files,
            desc="Merging",
            unit="Files",
            disable=not self.progress_bar,
        ):
            try:
                self.merger(input_path)
                n_merged += 1
            except CannotMerge as error:
                if not self.skip_broken_files:
                    raise
                self.log.warning("Skipping broken file: %s", error)

        self.log.info(
            "%d out of %d files have been merged!",
            n_merged,
            len(self.input_files),
        )

    def finish(self):
        # override activity meta with merge current activity
        current_activity = Provenance().current_activity.provenance
        self.merger.meta.activity = meta.Activity.from_provenance(current_activity)
        meta.write_to_hdf5(self.merger.meta.to_dict(), self.merger.h5file)


def main():
    tool = MergeTool()
    tool.run()


if __name__ == "main":
    main()
