"""
Create a working directory for ctapipe-process containing standard
configuration files.
"""
try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

import shutil

from ..core import Tool, Provenance
from ..core import traits
from ..version import __version__ as VERSION

CONFIGS_TO_WRITE = ["stage1_config.json", "stage2_config.json", "training_config.json"]

README_TEXT = f"""
ctapipe working directory
-------------------------

This working directory contains some example configuration files that are useful
for processing data with `ctapipe-process`. These include:

- stage1_config.json:  generate DL1 data from lower data levels
- stage2_config.json:  generate DL2 shower geometry from DL1 or lower levels
- training_config.json: generate both DL1 parameter and DL2 shower geometry data

You can modify these to change the output,  and run ctapipe using:

```
ctapipe-process --config <CONFIG> --input <EVENTS FILE>
``` 

Where <EVENTS FILE> is any ctapipe-readable event file at a lower or equal data 
level to the one requested to be produced.

Details about all configuration options can be found by running:

```
ctapipe-process --help-all
```

This file was generated using ctapipe version {VERSION}
"""


class QuickStartTool(Tool):
    """ Generate quick start files and directory structure """

    name = "ctapipe-quickstart"
    description = __doc__
    examples = """
    ctapipe-quickstart --workdir MyProduction
    """

    workdir = traits.Path(
        default_value="./Work",
        directory_ok=True,
        file_ok=False,
        help="working directory where configuration files should be written",
    ).tag(config=True)

    aliases = {("d", "workdir"): "QuickStartTool.workdir"}

    def setup(self):
        self.workdir.mkdir(parents=True, exist_ok=True)

    def start(self):
        for filename in CONFIGS_TO_WRITE:
            config = files("ctapipe.tools.tests.resources").joinpath(filename)
            destination = self.workdir / filename

            if destination.exists():
                self.log.warning(
                    "%s exists, please remove it if you want to generate a new one",
                    destination,
                )
                continue

            shutil.copy(config, destination)
            Provenance().add_output_file(destination, role="ctapipe-quickstart config")

        # also generate a README file
        readme = self.workdir / "README.md"
        if not readme.exists():
            with open(readme, "w") as outfile:
                outfile.write(README_TEXT)
            Provenance().add_output_file(readme, role="ctapipe-quickstart README")

    def finish(self):
        print(f"Generated examples in {self.workdir}")


def main():
    """ run the tool"""
    tool = QuickStartTool()
    tool.run()


if __name__ == "__main__":
    main()
