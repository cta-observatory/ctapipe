import ctapipe
import pkgutil
import importlib
from collections import defaultdict

from ctapipe.core import Component, Tool

ignore_traits = {
    "config",
    "parent",
    "argv",
    "log",
    "name",
    "subcommands",
    "version",
    "subapp",
    "aliases",
    "_loaded_config_files",
    "cli_config",
    "description",
    "examples",
    "extra_args",
    "flags",
    "keyvalue_description",
    "option_description",
    "raise_config_file_errors",
    "subcommand_description",
    "classes",
}


def test_all_traitlets_configurable():
    def find_all_traitlets(module, missing_config=None):
        module_name = module.__name__

        if missing_config is None:
            missing_config = defaultdict(set)

        for submodule_info in pkgutil.iter_modules(module.__path__):
            submodule = importlib.import_module(module_name + "." + submodule_info.name)

            if submodule_info.ispkg:
                find_all_traitlets(submodule, missing_config=missing_config)
            else:
                for obj_name in dir(submodule):
                    obj = getattr(submodule, obj_name)

                    try:
                        has_traits = issubclass(obj, (Component, Tool))
                    except TypeError:
                        has_traits = False

                    if has_traits:
                        for traitname, trait in obj.class_traits().items():
                            if (
                                not trait.metadata.get("config", False)
                                and traitname not in ignore_traits
                            ):
                                missing_config[f"{obj.__module__}.{obj.__name__}"].add(
                                    traitname
                                )

        return missing_config

    missing_config = find_all_traitlets(ctapipe)
    # if the test files, we need to know which failed
    for name, missing in missing_config.items():
        print("Class", name, "is missing .tag(config=True) for traitlets:", *missing)
    assert len(missing_config) == 0
