import pickle

from ctapipe.core.tool import Tool
from ctapipe.core.traits import Path
from ctapipe.io import TableLoader

from ..sklearn import Regressor



class TrainEnergyRegressor(Tool):

    output_path = Path(
        default_value=None,
        allow_none=False,
        directory_ok=False,
    ).tag(config=True)

    aliases = {
        ('i', 'input'): 'TableLoader.input_url',
        ('o', 'output'): 'TrainEnergyRegressor.output_path',
    }

    classes = [
        TableLoader,
        Regressor,
    ]

    def setup(self):
        ''''''
        self.loader = TableLoader(
            parent=self,
            load_dl1_images=False,
            load_dl1_parameters=True,
            load_dl2_geometry=True,
            load_simulated=True,
        )

    def start(self):
        self.models = {}
        for tel_type in self.loader.subarray.telescope_types:
            self.log.info("Training model for type %s", tel_type)
            self.models[tel_type] = Regressor(
                parent=self,
                target="true_energy",
                features=[
                    "hillas_intensity",
                    "hillas_width",
                    "hillas_length",
                    "morphology_num_pixels",
                    "concentration_cog",
                    "concentration_core",
                ]
            )
            table = self.loader.read_telescope_events([tel_type])
            self.models[tel_type].fit(table)

    def finish(self):
        with open(self.output_path, 'wb') as f:
            pickle.dump(self.models, f)



if __name__ == '__main__':
    TrainEnergyRegressor().run()
