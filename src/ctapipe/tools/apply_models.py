"""
Tool to apply machine learning models in bulk (as opposed to event by event).
"""

import numpy as np
import tables
from astropy.table import Table, join, vstack
from tqdm.auto import tqdm

from ctapipe.core.tool import Tool
from ctapipe.core.traits import Bool, Integer, List, Path, classes_with_traits, flag
from ctapipe.io import HDF5Merger, TableLoader, write_table
from ctapipe.io.astropy_helpers import join_allow_empty, read_table
from ctapipe.io.tableio import TelListToMaskTransform
from ctapipe.reco import Reconstructor

__all__ = [
    "ApplyModels",
]


class ApplyModels(Tool):
    """Apply machine learning models to data.

    This tool predicts all events at once. To apply models in the
    regular event loop, set the appropriate options to ``ctapipe-process``.

    Models need to be trained with
    `~ctapipe.tools.train_energy_regressor.TrainEnergyRegressor`
    and
    `~ctapipe.tools.train_particle_classifier.TrainParticleClassifier`.
    """

    name = "ctapipe-apply-models"
    description = __doc__
    examples = """
    ctapipe-apply-models \\
        --input gamma.dl2.h5 \\
        --reconstructor energy_regressor.pkl \\
        --reconstructor particle-classifier.pkl \\
        --output gamma_applied.dl2.h5
    """

    input_url = Path(
        default_value=None,
        allow_none=False,
        directory_ok=False,
        exists=True,
        help="Input dl1b/dl2 file",
    ).tag(config=True)

    output_path = Path(
        default_value=None,
        allow_none=False,
        directory_ok=False,
        help="Output file",
    ).tag(config=True)

    reconstructor_paths = List(
        Path(exists=True, directory_ok=False),
        default_value=[],
        help="Paths to trained reconstructors to be applied to the input data",
    ).tag(config=True)

    chunk_size = Integer(
        default_value=100000,
        allow_none=True,
        help="How many subarray events to load at once for making predictions.",
    ).tag(config=True)

    n_jobs = Integer(
        default_value=None,
        allow_none=True,
        help="Number of threads to use for the reconstruction. This overwrites the values in the config",
    ).tag(config=True)

    progress_bar = Bool(
        help="show progress bar during processing",
        default_value=True,
    ).tag(config=True)

    aliases = {
        ("i", "input"): "ApplyModels.input_url",
        ("r", "reconstructor"): "ApplyModels.reconstructor_paths",
        ("o", "output"): "ApplyModels.output_path",
        "n-jobs": "ApplyModels.n_jobs",
        "chunk-size": "ApplyModels.chunk_size",
    }

    flags = {
        **flag(
            "progress",
            "ProcessorTool.progress_bar",
            "show a progress bar during event processing",
            "don't show a progress bar during event processing",
        ),
        **flag(
            "dl1-parameters",
            "HDF5Merger.dl1_parameters",
            "Include dl1 parameters",
            "Exclude dl1 parameters",
        ),
        **flag(
            "dl1-images",
            "HDF5Merger.dl1_images",
            "Include dl1 images",
            "Exclude dl1 images",
        ),
        **flag(
            "true-parameters",
            "HDF5Merger.true_parameters",
            "Include true parameters",
            "Exclude true parameters",
        ),
        **flag(
            "true-images",
            "HDF5Merger.true_images",
            "Include true images",
            "Exclude true images",
        ),
        "overwrite": (
            {
                "HDF5Merger": {"overwrite": True},
                "ApplyModels": {"overwrite": True},
            },
            "Overwrite output file if it exists",
        ),
    }

    classes = [TableLoader] + classes_with_traits(Reconstructor)

    def setup(self):
        """
        Initialize components from config
        """
        self.check_output(self.output_path)
        self.log.info("Copying to output destination.")

        with HDF5Merger(self.output_path, parent=self) as merger:
            merger(self.input_url)

        self.h5file = self.enter_context(tables.open_file(self.output_path, mode="r+"))
        self.loader = self.enter_context(
            TableLoader(
                self.input_url,
                parent=self,
            )
        )

        self._reconstructors = []
        for path in self.reconstructor_paths:
            r = Reconstructor.read(path, parent=self, subarray=self.loader.subarray)
            if self.n_jobs:
                r.n_jobs = self.n_jobs
            self._reconstructors.append(r)

    def start(self):
        """Apply models to input tables"""
        chunk_iterator = self.loader.read_telescope_events_by_id_chunked(
            self.chunk_size,
            simulated=False,
            true_parameters=False,
            observation_info=True,
            instrument=True,
        )
        bar = tqdm(
            chunk_iterator,
            desc="Applying reconstructors",
            unit=" Array Events",
            total=chunk_iterator.n_total,
            disable=not self.progress_bar,
        )
        with bar:
            for chunk, (start, stop, tel_tables) in enumerate(chunk_iterator):
                for reconstructor in self._reconstructors:
                    self.log.debug("Applying %s to chunk %d", reconstructor, chunk)
                    self._apply(reconstructor, tel_tables, start=start, stop=stop)

                bar.update(stop - start)

    def _apply(self, reconstructor, tel_tables, start, stop):
        prefix = reconstructor.prefix

        for tel_id, table in tel_tables.items():
            tel = self.loader.subarray.tel[tel_id]

            if len(table) == 0:
                self.log.info("No events for telescope %d", tel_id)
                continue

            try:
                predictions = reconstructor.predict_table(tel, table)
            except KeyError:
                self.log.warning(
                    "No model in %s for telescope type %s, skipping tel %d",
                    reconstructor,
                    tel,
                    tel_id,
                )
                continue

            for prop, prediction_table in predictions.items():
                # copy/overwrite columns into full feature table
                new_columns = prediction_table.colnames
                for col in new_columns:
                    table[col] = prediction_table[col]

                output_columns = ["obs_id", "event_id", "tel_id"] + new_columns
                write_table(
                    table[output_columns],
                    self.output_path,
                    f"/dl2/event/telescope/{prop}/{prefix}/tel_{tel_id:03d}",
                    append=True,
                )

        self._combine(reconstructor, tel_tables, start=start, stop=stop)

    def _combine(self, reconstructor, tel_tables, start, stop):
        stereo_table = vstack(list(tel_tables.values()))

        # stacking the single telescope tables and joining
        # potentially changes the order of the subarray events.
        # to ensure events are stored in the correct order,
        # we resort to trigger table order
        trigger = read_table(
            self.h5file, "/dl1/event/subarray/trigger", start=start, stop=stop
        )[["obs_id", "event_id"]]
        trigger["__sort_index__"] = np.arange(len(trigger))

        stereo_table = join_allow_empty(
            stereo_table,
            trigger,
            keys=["obs_id", "event_id"],
            join_type="left",
        )
        stereo_table.sort("__sort_index__")

        combiner = reconstructor.stereo_combiner
        stereo_predictions = combiner.predict_table(stereo_table)
        del stereo_table

        trafo = TelListToMaskTransform(self.loader.subarray)
        for c in filter(
            lambda c: c.name.endswith("telescopes"),
            stereo_predictions.columns.values(),
        ):
            stereo_predictions[c.name] = np.array([trafo(r) for r in c])
            stereo_predictions[c.name].description = c.description

        write_table(
            stereo_predictions,
            self.output_path,
            f"/dl2/event/subarray/{combiner.property}/{combiner.prefix}",
            append=True,
        )

        for tel_table in tel_tables.values():
            _add_stereo_prediction(tel_table, stereo_predictions)


def _add_stereo_prediction(tel_events, array_events):
    """Add columns from array_events table to tel_events table"""
    join_table = Table(
        {
            "obs_id": tel_events["obs_id"],
            "event_id": tel_events["event_id"],
            "__sort_index__": np.arange(len(tel_events)),
        }
    )
    joined = join(join_table, array_events, keys=["obs_id", "event_id"])
    del join_table
    joined.sort("__sort_index__")
    joined.remove_columns(["obs_id", "event_id", "__sort_index__"])
    for colname in joined.colnames:
        tel_events[colname] = joined[colname]


def main():
    ApplyModels().run()


if __name__ == "__main__":
    main()
