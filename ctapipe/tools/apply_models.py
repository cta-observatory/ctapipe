"""
Tool to apply machine learning models in bulk (as opposed to event by event).
"""
import shutil

import numpy as np
import tables
from astropy.table.operations import hstack, vstack
from tqdm.auto import tqdm

from ctapipe.core.tool import Tool
from ctapipe.core.traits import Bool, Path, flag
from ctapipe.io import TableLoader, write_table
from ctapipe.io.tableio import TelListToMaskTransform
from ctapipe.reco import EnergyRegressor, ParticleClassifier, StereoCombiner

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

    overwrite = Bool(default_value=False).tag(config=True)

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

    energy_regressor_path = Path(
        default_value=None,
        allow_none=True,
        exists=True,
        directory_ok=False,
        help="Input path for the trained EnergyRegressor",
    ).tag(config=True)

    particle_classifier_path = Path(
        default_value=None,
        allow_none=True,
        exists=True,
        directory_ok=False,
        help="Input path for the trained ParticleClassifier",
    ).tag(config=True)

    aliases = {
        ("i", "input"): "ApplyModels.input_url",
        "energy-regressor": "ApplyModels.energy_regressor_path",
        "particle-classifier": "ApplyModels.particle_classifier_path",
        ("o", "output"): "ApplyModels.output_path",
    }

    flags = {
        **flag(
            "overwrite",
            "ApplyModels.overwrite",
            "Overwrite tables in output file if it exists",
            "Don't overwrite tables in output file if it exists",
        ),
        "f": (
            {"ApplyModels": {"overwrite": True}},
            "Overwrite output file if it exists",
        ),
    }

    classes = [
        TableLoader,
        EnergyRegressor,
        ParticleClassifier,
        StereoCombiner,
    ]

    def setup(self):
        """
        Initialize components from config
        """
        self.log.info("Copying to output destination.")
        shutil.copy(self.input_url, self.output_path)

        self.h5file = tables.open_file(self.output_path, mode="r+")
        self.loader = TableLoader(
            parent=self,
            h5file=self.h5file,
            load_dl1_parameters=True,
            load_dl2=True,
            load_instrument=True,
            load_dl1_images=False,
            load_simulated=False,
        )

        self._reconstructors = []

        if self.energy_regressor_path is not None:
            self._reconstructors.append(
                EnergyRegressor.read(
                    self.energy_regressor_path,
                    parent=self,
                )
            )
        if self.particle_classifier_path is not None:
            self._reconstructors.append(
                ParticleClassifier.read(
                    self.particle_classifier_path,
                    parent=self,
                )
            )

    def start(self):
        """Apply models to input tables"""
        for reconstructor in self._reconstructors:
            self.log.info("Applying %s to telescope events", reconstructor)
            mono_predictions = self._apply(reconstructor)
            self.log.info(
                "Combining telescope-wise predictions of %s to array event prediction",
                reconstructor,
            )
            self._combine(reconstructor.stereo_combiner, mono_predictions)
            # FIXME: this is a not-so-nice solution for the issues that
            # the table loader does not seem to see the newly written tables
            # we close and reopen the file and then table loader loads also the new tables
            self.h5file.close()
            self.h5file = tables.open_file(self.output_path, mode="r+")
            self.loader.h5file = self.h5file

    def _apply(self, reconstructor):
        prefix = reconstructor.model_cls
        property = reconstructor.property

        tel_tables = []

        desc = f"Applying {reconstructor.__class__.__name__}"
        unit = "telescope"
        for tel_id, tel in tqdm(self.loader.subarray.tel.items(), desc=desc, unit=unit):
            if tel not in reconstructor._models:
                self.log.warning(
                    "No model in %s for telescope type %s, skipping tel %d",
                    reconstructor,
                    tel,
                    tel_id,
                )
                continue

            table = self.loader.read_telescope_events([tel_id])
            if len(table) == 0:
                self.log.warning("No events for telescope %d", tel_id)
                continue

            table.remove_columns([c for c in table.colnames if c.startswith(prefix)])

            predictions = reconstructor.predict_table(tel, table)
            table = hstack(
                [table, predictions],
                join_type="exact",
                metadata_conflicts="ignore",
            )
            write_table(
                table[["obs_id", "event_id", "tel_id"] + predictions.colnames],
                self.output_path,
                f"/dl2/event/telescope/{property}/{prefix}/tel_{tel_id:03d}",
                mode="a",
                overwrite=self.overwrite,
            )
            tel_tables.append(table)

        if len(tel_tables) == 0:
            raise ValueError("No predictions made for any telescope")

        return vstack(tel_tables)

    def _combine(self, combiner, mono_predictions):
        stereo_predictions = combiner.predict_table(mono_predictions)

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
            mode="a",
            overwrite=self.overwrite,
        )

    def finish(self):
        """Close input file"""
        self.h5file.close()


def main():
    ApplyModels().run()


if __name__ == "__main__":
    main()
