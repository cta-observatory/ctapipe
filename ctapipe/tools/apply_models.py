"""
Tool to apply machine learning models in bulk (as opposed to event by event).
"""
import shutil

import numpy as np
import tables
from astropy.table.operations import vstack
from tqdm.auto import tqdm

from ctapipe.core.tool import Tool
from ctapipe.core.traits import Integer, List, Path
from ctapipe.io import TableLoader, write_table
from ctapipe.io.astropy_helpers import read_table
from ctapipe.io.tableio import TelListToMaskTransform
from ctapipe.io.tableloader import _join_subarray_events
from ctapipe.reco import (
    DispReconstructor,
    EnergyRegressor,
    ParticleClassifier,
    StereoCombiner,
)
from ctapipe.reco.reconstructor import Reconstructor

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
        --energy-regressor energy_regressor.pkl \\
        --particle-classifier particle-classifier.pkl \\
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

    aliases = {
        ("i", "input"): "ApplyModels.input_url",
        ("r", "reconstructor"): "ApplyModels.reconstructor_paths",
        ("o", "output"): "ApplyModels.output_path",
        "chunk-size": "ApplyModels.chunk_size",
    }

    classes = [
        TableLoader,
        EnergyRegressor,
        ParticleClassifier,
        DispReconstructor,
        StereoCombiner,
    ]

    def setup(self):
        """
        Initialize components from config
        """
        self.check_output(self.output_path)
        self.log.info("Copying to output destination.")
        shutil.copy(self.input_url, self.output_path)

        self.h5file = self.enter_context(tables.open_file(self.output_path, mode="r+"))
        self.loader = TableLoader(
            parent=self,
            h5file=self.h5file,
            load_dl1_parameters=True,
            load_dl2=True,
            load_instrument=True,
            load_dl1_images=False,
            load_simulated=False,
            load_observation_info=True,
        )

        self._reconstructors = [
            Reconstructor.read(path) for path in self.reconstructor_paths
        ]

    def start(self):
        """Apply models to input tables"""
        for reconstructor in self._reconstructors:
            self.log.info("Applying %s", reconstructor)
            self._apply(reconstructor)
            # FIXME: this is a not-so-nice solution for the issues that
            # the table loader does not seem to see the newly written tables
            # we close and reopen the file and then table loader loads also the new tables
            self.h5file.close()
            self.h5file = self.enter_context(
                tables.open_file(self.output_path, mode="r+")
            )
            self.loader.h5file = self.h5file

    def _apply(self, reconstructor):
        prefix = reconstructor.prefix

        desc = f"Applying {reconstructor.__class__.__name__}"
        unit = "chunk"

        chunk_iterator = self.loader.read_telescope_events_by_id_chunked(
            self.chunk_size
        )
        for chunk in tqdm(chunk_iterator, desc=desc, unit=unit):
            tel_tables = []

            for tel_id, table in chunk.items():
                tel = self.loader.subarray.tel[tel_id]
                if tel not in reconstructor._models:
                    self.log.warning(
                        "No model in %s for telescope type %s, skipping tel %d",
                        reconstructor,
                        tel,
                        tel_id,
                    )
                    continue

                if len(table) == 0:
                    self.log.warning("No events for telescope %d", tel_id)
                    continue

                table.remove_columns(
                    [c for c in table.colnames if c.startswith(prefix)]
                )

                predictions = reconstructor.predict_table(tel, table)

                for prop, prediction_table in predictions.items():
                    new_columns = prediction_table.colnames
                    output_columns = ["obs_id", "event_id", "tel_id"] + new_columns

                    # copy columns into full feature table
                    for col in new_columns:
                        table[col] = prediction_table[col]

                    write_table(
                        table[output_columns],
                        self.output_path,
                        f"/dl2/event/telescope/{prop}/{prefix}/tel_{tel_id:03d}",
                        append=True,
                    )
                tel_tables.append(table)

            if len(tel_tables) == 0:
                raise ValueError("No predictions made for any telescope")

            self._combine(
                reconstructor.stereo_combiner,
                vstack(tel_tables),
                start=chunk_iterator.start,
                stop=chunk_iterator.stop,
            )

    def _combine(self, combiner, mono_predictions, start=None, stop=None):
        stereo_predictions = combiner.predict_table(mono_predictions)

        trafo = TelListToMaskTransform(self.loader.subarray)
        for c in filter(
            lambda c: c.name.endswith("telescopes"),
            stereo_predictions.columns.values(),
        ):
            stereo_predictions[c.name] = np.array([trafo(r) for r in c])
            stereo_predictions[c.name].description = c.description

        # to ensure events are stored in the correct order,
        # we resort to trigger table order
        trigger = read_table(
            self.h5file, "/dl1/event/subarray/trigger", start=start, stop=stop
        )[["obs_id", "event_id"]]
        trigger["__sort_index__"] = np.arange(len(trigger))
        stereo_predictions = _join_subarray_events(trigger, stereo_predictions)
        stereo_predictions.sort("__sort_index__")
        del stereo_predictions["__sort_index__"]

        write_table(
            stereo_predictions,
            self.output_path,
            f"/dl2/event/subarray/{combiner.property}/{combiner.prefix}",
            append=True,
        )

    def finish(self):
        """Close input file"""
        self.h5file.close()


def main():
    ApplyModels().run()


if __name__ == "__main__":
    main()
