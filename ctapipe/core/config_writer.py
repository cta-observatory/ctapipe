import logging
import textwrap

import traitlets

log = logging.getLogger(__name__)


def trait_dict_to_yaml(conf, conf_repr="", indent_level=0):
    """
    Using a dictionnary of traits, will write this configuration to file. Each value is either a subsection or a
    trait object so that we can extract value, default value and help message

    :param conf: Dictionnary of traits. Architecture reflect what is needed in the yaml file.
    :param str conf_repr: internal variable used for recursivity. You shouldn't use that parameter
    :param int indent_level: internal variable used for recursivity. You shouldn't use that parameter

    :return: str representation of conf, ready to store in a .yaml file
    """
    indent_str = "  "

    for k, v in conf.items():
        if isinstance(v, dict):
            conf_repr += f"\n{indent_str * indent_level}{k}:\n"
            conf_repr = trait_dict_to_yaml(v, conf_repr, indent_level=indent_level + 1)
        else:
            conf_repr += _trait_to_str(v, indent_level=indent_level)

    return conf_repr


def _trait_to_str(trait, help=True, indent_level=0):
    """
    Represent a trait in a futur yaml file, given prior information on its position.

    :param traitlets.trait trait:
    :param bool help: [optional] True by default
    :param indent_level: Indentation level to apply to the trait when creating the string, for correct display in
    parent string.

    :return: String representation of the input trait.
    :rtype: str
    """
    indent_str = "  "

    def commented(text, indent_level=indent_level, width=144):
        """return a commented, wrapped block."""
        return textwrap.fill(
            text,
            width=width,
            initial_indent=indent_str * indent_level + "# ",
            subsequent_indent=indent_str * indent_level + "# ",
        )

    trait_repr = "\n"

    if help:
        h_msg = trait.help

        if h_msg:
            trait_repr += f"{commented(h_msg, indent_level=indent_level)}\n"

    trait_value = trait.get_default_value()
    # add quotes for strings
    if isinstance(trait, traitlets.Unicode):
        trait_value = f"'{trait_value}'"

    trait_repr += f"{indent_str*indent_level}{trait.name}: {trait_value}\n"

    return trait_repr
