"""
Merge DL1-files from ctapipe-process tool
"""
import sys
from argparse import ArgumentParser
from pathlib import Path

from tqdm.auto import tqdm
from traitlets import List

from ..core import Provenance, Tool, traits
from ..core.traits import Bool, Unicode, flag
from ..io.select_merge_hdf5 import HDF5Merger


class MergeTool(Tool):
    name = "ctapipe-merge"
    description = "Merges DL1-files created by the stage1-tool"
    examples = """
    To merge DL1-files created by the stage1-tool from current directory:

    > ctapipe-merge file1.h5 file2.h5 file3.h5 --output=/path/output_file.h5 --progress

    For merging files from a specific directory with a specific pattern, use:

    > ctapipe-merge --input-dir=/input/dir/ --output=/path/output_file.h5 --progress
    --pattern='*.dl1.h5'

    If no pattern is given, all .h5 files of the given directory will be taken as input.
    """
    input_dir = traits.Path(
        default_value=None,
        help="Input dl1-directory",
        allow_none=True,
        exists=True,
        directory_ok=True,
        file_ok=False,
    ).tag(config=True)

    input_files = List(
        traits.Path(exists=True, directory_ok=False),
        default_value=[],
        help="Input dl1-files",
    ).tag(config=True)

    progress_bar = Bool(
        False,
        help="Show progress bar during processing",
    ).tag(config=True)

    file_pattern = Unicode(
        default_value="*.h5", help="Give a specific file pattern for the input files"
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
            "HDF5Merger.dl2-subarray",
            "Include dl2 subarray-wise data",
            "Exclude dl2 subarray-wise data",
        ),
    }

    classes = [HDF5Merger]

    def setup(self):
        self.merger = HDF5Merger(parent=self)

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

    def start(self):
        n_merged = 0

        for input_path in tqdm(
            self.input_files,
            desc="Merging",
            unit="Files",
            disable=not self.progress_bar,
        ):
            self.merger(input_path)
            n_merged += 1

        self.log.info(
            "%d out of %d files have been merged!",
            n_merged,
            len(self.input_files),
        )

    def finish(self):
        # overide activity meta with merge current activity
        current_activity = Provenance().current_activity.provenance
        self.merger.meta.activity = meta.Activity.from_provenance(current_activity)
        meta.write_to_hdf5(self.merger.meta.to_dict(), self.merger.h5file)
        self.merger.close()


def main():
    tool = MergeTool()
    tool.run()


if __name__ == "main":
    main()
