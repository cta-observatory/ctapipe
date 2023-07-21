import logging
import re
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
            # Separate this new block from previous content
            conf_repr += "\n"

            # Add summary line from class docstring
            class_help = v.pop(
                "__doc__"
            )  # Pop to avoid treating this key:value as a parameter later on.
            conf_repr += f"{wrap_comment(class_help, indent_level=indent_level)}\n"

            conf_repr += f"{indent_str * indent_level}{k}:\n"
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

    trait_repr = "\n"

    trait_type = get_trait_type(trait)

    # By default, help message only have info about parameter type
    h_msg = f"[{trait_type}] "

    if "Enum" in trait_type:
        enum_values = get_enum_values(trait)
        h_msg += f"(Possible values: {enum_values}) "

    if help:
        h_msg += trait.help

    # Get rid of unnecessary formatting because we'll redo that
    h_msg = clean_help_msg(h_msg)

    trait_repr += f"{wrap_comment(h_msg, indent_level=indent_level)}\n"

    trait_value = trait.default()

    # add quotes for strings
    if isinstance(trait, traitlets.Unicode):
        trait_value = f"'{trait_value}'"

    # Make sure that list of values end up on multiple lines (In particular quality criteria)
    value_repr = ""
    if isinstance(trait_value, list):
        for v in trait_value:
            # tuples are not correctly handled by yaml, so we convert them to list here. They'll be converted to tuple
            # when reading again.
            if isinstance(v, tuple):
                v = list(v)
            value_repr += f"\n{indent_str * indent_level}- {v}"
    else:
        value_repr += f"{trait_value}"

    # Automatically comment all parameters that are unvalid
    commented = ""
    if trait_value == traitlets.Undefined:
        commented = "#"

    trait_repr += f"{indent_str*indent_level}{commented}{trait.name}: {value_repr}\n"

    return trait_repr


def get_trait_type(trait):
    """
    Get trait type. If needed, use recursion for sub-types in case of list, set...

    :param traitlets.trait trait: Input trait

    :return: str representation of the trait type
    :rtype: str
    """
    _repr = f"{trait.__class__.__name__}"

    if hasattr(trait, "_trait"):
        _repr += f"({get_trait_type(trait._trait)})"

    return _repr


def get_enum_values(trait):
    """
    Get possible values for a trait with Enum. Note that this can also be a list of enum.

    We do not test the trait, we assume that 'trait' is either Enum or has a sub-trait with Enum in it.
    If needed, use recursion for sub-types in case of list, set...

    :param traitlets.trait trait: Input trait

    :return: List of possible values for the Enum in this trait
    :rtype: str
    """
    trait_type = trait.__class__.__name__

    if trait_type in ["Enum", "CaselessStrEnum"]:
        values = list(
            trait.values
        )  # Sometimes, the output is not a list (e.g. norm_cls)
    elif trait_type == "UseEnum":
        values = trait.enum_class._member_names_
    else:
        try:
            values = get_enum_values(trait._trait)
        except AttributeError:
            log.error(f"Can't find an Enum type in trait '{trait.name}'")
            raise

    return values


def get_summary_doc(cls):
    """
    Applied on a class object, will retrieve the first line of the docstring.

    :param obj cls:

    :return: Summary line from input docstring
    :rtype: str
    """
    first_line = cls.__doc__.split("\n\n")[0]

    first_line = clean_help_msg(first_line)

    return first_line


def clean_help_msg(msg):
    """
    Clean and merge lines in a string to have only one line, get rid of tabulation and extra spaces.

    :param str msg:
    :return: cleaned string
    """
    # Merge all lines, including tabulation if need be
    msg = re.sub("\n *", " ", msg)

    # clean extra spaces (regexp tend to leave a starting space because there's usually a newline at the start)
    msg = msg.strip()

    return msg


def get_default_config(cls):
    """
    Get list of all traits from that class.

    This is intented to be used as a class methods for all included class a Tool might have. Since the method is
    always the same, for the sake of maintainability, We'll just use that function.

    :param cls:
    :return:
    """
    conf = {cls.__name__: cls.traits(cls, config=True)}

    # Add class doc for later use
    conf[cls.__name__]["__doc__"] = get_summary_doc(cls)

    return conf


def wrap_comment(text, indent_level=0, width=144, indent_str="  "):
    """return a commented, wrapped block."""
    return textwrap.fill(
        text,
        width=width,
        initial_indent=indent_str * indent_level + "# ",
        subsequent_indent=indent_str * indent_level + "# ",
    )
