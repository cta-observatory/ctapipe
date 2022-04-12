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

    def setup(self):
        ''''''
        self.loader = TableLoader(
            parent=self, 
            load_dl1_images=False,
            load_dl1_parameters=True,
            load_dl2_geometry=True,
            load_simulated=True,
        )
        self.model = Regressor(
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

    def start(self):
        table = self.loader.read_telescope_events()
        self.model.fit(table)

    def finish(self):
        self.model.write(self.output_path)




if __name__ == '__main__':
    TrainEnergyRegressor().run()
