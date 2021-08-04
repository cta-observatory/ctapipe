"""
Create a working directory for ctapipe-process containing standard
configuration files.
"""
try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

from pathlib import Path

from ..core import Provenance, Tool, traits
from ..version import __version__ as VERSION

__all__ = ["QuickStartTool"]

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


def copy_with_transforms(input_file: Path, output_file: Path, transforms: dict):
    """reads input_file and writes output_file, swapping text listed in the
    transformations dict

    Parameters
    ----------
    input_file: str
        template file to read
    output_file: str
        file to write
    transformations: Dict[str, str]
        dict of search and replacement strings
    """

    input_file = Path(input_file)
    output_file = Path(output_file)

    template = input_file.read_text()
    for find, replace in transforms.items():
        template = template.replace(find, replace)

    output_file.write_text(template)


class QuickStartTool(Tool):
    """
    Generate quick start files and directory structure.
    """

    name = "ctapipe-quickstart"
    description = __doc__
    examples = """
    ctapipe-quickstart --workdir MyProduction

    ctapipe-quickstart --name "my name" --contact "me@thing.com" --workdir Work
    """

    workdir = traits.Path(
        default_value="./Work",
        directory_ok=True,
        file_ok=False,
        help="working directory where configuration files should be written",
    ).tag(config=True)

    contact_name = traits.Unicode("", help="Contact name").tag(config=True)
    contact_email = traits.Unicode("", help="Contact email").tag(config=True)
    contact_organization = traits.Unicode("", help="Contact organization").tag(
        config=True
    )

    aliases = {
        ("d", "workdir"): "QuickStartTool.workdir",
        ("n", "name"): "QuickStartTool.contact_name",
        ("e", "email"): "QuickStartTool.contact_email",
        ("o", "org"): "QuickStartTool.contact_organization",
    }

    def setup(self):
        self.workdir.mkdir(parents=True, exist_ok=True)

        if self.contact_name == "":
            print("Enter your contact name: ", end="")
            self.contact_name = input()

        if self.contact_email == "":
            print("Enter your contact email: ", end="")
            self.contact_email = input()

        if self.contact_organization == "":
            print("Enter your organization: ", end="")
            self.contact_organization = input()

        self.transforms = {
            "YOUR-NAME-HERE": self.contact_name,
            "YOUREMAIL@EXAMPLE.ORG": self.contact_email,
            "YOUR-ORGANIZATION": self.contact_organization,
        }

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

            copy_with_transforms(config, destination, transforms=self.transforms)
            Provenance().add_output_file(destination, role="ctapipe-process config")

        # also generate a README file
        readme = self.workdir / "README.md"
        if not readme.exists():
            readme.write_text(README_TEXT)
            Provenance().add_output_file(readme, role="README")

    def finish(self):
        print(f"Generated examples in {self.workdir}")


def main():
    """ run the tool"""
    tool = QuickStartTool()
    tool.run()


if __name__ == "__main__":
    main()
