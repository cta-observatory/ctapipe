"""
common tool for ``ctapipe train XXX``
"""

from traitlets.config import Application

from ..version import __version__


class TrainTool(Application):
    """Main entry point for training tools"""

    name = "ctapipe train"
    version = __version__

    subcommands = {
        "energy-regressor": (
            "ctapipe.tools.train_energy_regressor.TrainEnergyRegressor",
            "Train telescope-type-wise energy regression models",
        ),
        "particle-classifier": (
            "ctapipe.tools.train_particle_classifier.TrainParticleClassifier",
            "Train telescope-type-wise binary classficiation models",
        ),
        "disp-regressor": (
            "ctapipe.tools.train_disp_regressor.TrainDispRegressor",
            "Train telescope-type-wise disp regression models\n"
            "for direction reconstruction",
        ),
    }

    def run(self):
        if self.subapp is None:
            if len(self.extra_args) > 0:
                print(f"Unknown sub-command {self.extra_args[0]}\n\n")
            self.print_subcommands()
            self.exit(1)
        breakpoint()
        self.subapp.run()

    def write_provenance(self):
        """This tool should not write any provenance log"""


def main():
    TrainTool.launch_instance()


if __name__ == "__main__":
    main()
