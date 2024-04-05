"""
This module defines classes to enable per-telescope configuration of trait values
"""
import copy
import logging
from collections import UserList
from fnmatch import fnmatch

import numpy as np
from traitlets import List, TraitError, TraitType, Undefined

from .component import Component

__all__ = [
    "TelescopeComponent",
    "TelescopeParameter",
    "TelescopeParameterLookup",
    "TelescopePatternList",
]


logger = logging.getLogger(__name__)


class TelescopeComponent(Component):
    """
    A component that needs a `~ctapipe.instrument.SubarrayDescription` to be constructed,
    and which contains configurable `~ctapipe.core.telescope_component.TelescopeParameter`
    fields that must be configured on construction.
    """

    def __init__(self, subarray, config=None, parent=None, **kwargs):
        super().__init__(config, parent, **kwargs)
        self.subarray = subarray

    @property
    def subarray(self):
        return self._subarray

    @subarray.setter
    def subarray(self, subarray):
        self._subarray = subarray
        # configure all of the TelescopeParameters
        for attr, trait in self.class_traits().items():
            if not isinstance(trait, TelescopeParameter):
                continue

            # trait is the TelescopeParameter descriptor at the class,
            # need to get the value at the instance, which will be a TelescopePatternList
            pattern_list = getattr(self, attr)
            if pattern_list is not None:
                pattern_list.attach_subarray(subarray)

    @classmethod
    def from_name(cls, name, subarray, config=None, parent=None, **kwargs):
        """
        Obtain an instance of a subclass via its name

        Parameters
        ----------
        name : str
            Name of the subclass to obtain
        subarray: ctapipe.instrument.SubarrayDescription
            The current subarray for this TelescopeComponent.
        config : traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            This argument is typically only specified when using this method
            from within a Tool.
        parent : ctapipe.core.Tool
            Tool executable that is calling this component.
            Passes the correct logger and configuration to the component.
            This argument is typically only specified when using this method
            from within a Tool (config need not be passed if parent is used).
        kwargs
            Are passed to the subclass

        Returns
        -------
        Instance
            Instance of subclass to this class
        """
        requested_subclass = cls.non_abstract_subclasses()[name]
        return requested_subclass(
            subarray=subarray, config=config, parent=parent, **kwargs
        )


class TelescopePatternList(UserList):
    """
    Representation for a list of telescope pattern tuples. This is a helper class
    used  by the Trait TelescopeParameter as its value type
    """

    def __init__(self, *args):
        super().__init__(*args)
        self._lookup = None
        self._subarray = None

        for i in range(len(self)):
            self[i] = self.single_to_pattern(self[i])

    @property
    def tel(self):
        """access the value per telescope_id, e.g. ``param.tel[2]``"""
        if self._lookup:
            return self._lookup
        else:
            raise RuntimeError(
                "No TelescopeParameterLookup was registered. You must "
                "call attach_subarray() first"
            )

    @staticmethod
    def single_to_pattern(value):
        # make sure we only change things that are not already a
        # pattern tuple
        if (
            not isinstance(value, tuple)
            or len(value) != 3
            or value[0] not in {"type", "id"}
        ):
            return ["type", "*", value]

        return value

    def append(self, value):
        """Validate and then append a new value"""
        super().append(self.single_to_pattern(value))

    def attach_subarray(self, subarray):
        """
        Register a SubarrayDescription so that the user-specified values can be
        looked up by tel_id. This must be done before using the ``.tel[x]`` property
        """
        self._subarray = subarray
        self._lookup.attach_subarray(subarray)


class TelescopeParameterLookup:
    def __init__(self, telescope_parameter_list):
        """
        Handles the lookup of corresponding configuration value from a list of
        tuples for a tel_id.

        Parameters
        ----------
        telescope_parameter_list : list
            List of tuples in the form `[(command, argument, value), ...]`
        """
        self._telescope_parameter_list = copy.deepcopy(telescope_parameter_list)
        self._value_for_tel_id = None
        self._value_for_type = None
        self._subarray = None
        self._subarray_global_value = Undefined
        self._type_strs = None
        for param in telescope_parameter_list:
            if param[1] == "*":
                self._subarray_global_value = param[2]

    def attach_subarray(self, subarray):
        """
        Prepare the TelescopeParameter by informing it of the
        subarray description

        Parameters
        ----------
        subarray: ctapipe.instrument.SubarrayDescription
            Description of the subarray
            (includes mapping of tel_id to tel_type)
        """
        self._subarray = subarray
        self._value_for_tel_id = {}
        self._value_for_type = {}
        self._type_strs = {str(tel) for tel in self._subarray.telescope_types}
        for command, arg, value in self._telescope_parameter_list:
            if command == "type":
                matched_tel_types = [
                    str(t) for t in subarray.telescope_types if fnmatch(str(t), arg)
                ]
                logger.debug(f"argument '{arg}' matched: {matched_tel_types}")

                if len(matched_tel_types) == 0:
                    logger.warning(
                        "TelescopeParameter type argument '%s' did not match "
                        "any known telescope types",
                        arg,
                    )

                for tel_type in matched_tel_types:
                    self._value_for_type[tel_type] = value
                    for tel_id in subarray.get_tel_ids_for_type(tel_type):
                        self._value_for_tel_id[tel_id] = value
            elif command == "id":
                self._value_for_tel_id[int(arg)] = value
            else:
                raise ValueError(f"Unrecognized command: {command}")

    def __getitem__(self, tel: int | str | None):
        """
        Returns the resolved parameter for the given telescope id
        """
        if tel is None:
            if self._subarray_global_value is not Undefined:
                return self._subarray_global_value

            raise KeyError("No subarray global value set for TelescopeParameter")

        if self._value_for_tel_id is None:
            raise ValueError(
                "TelescopeParameterLookup: No subarray attached, call "
                "`attach_subarray` first before trying to access a value by tel_id"
            )

        if isinstance(tel, int | np.integer):
            try:
                return self._value_for_tel_id[tel]
            except KeyError:
                if tel not in self._subarray.tel:
                    raise KeyError(f"No telescope with id {tel} in subarray")
                raise KeyError(
                    "TelescopeParameterLookup: no "
                    f"parameter value was set for telescope with tel_id="
                    f"{tel}. Please set it explicitly, "
                    f"or by telescope type or '*'."
                )

        from ctapipe.instrument import TelescopeDescription

        if isinstance(tel, TelescopeDescription):
            tel = str(tel)

        if isinstance(tel, str):
            if tel not in self._type_strs:
                raise ValueError(
                    f"Unknown telescope type {tel}, known: {self._type_strs}"
                )
            try:
                return self._value_for_type[tel]
            except KeyError:
                raise KeyError(
                    "TelescopeParameterLookup: no "
                    "parameter value was set for telescope type"
                    f" '{tel}'. Please set explicitly or using '*'."
                )

        raise TypeError(f"Unsupported lookup type: {type(tel)}")


class TelescopeParameter(List):
    """
    Allow a parameter value to be specified as a simple value (of type *dtype*),
    or as a list of patterns that match different telescopes.

    The patterns are given as a list of 3-tuples in the
    form: ``[(command, argument, value), ...]``.

    Command can be one of:

    - ``'type'``: argument is then a telescope type  string (e.g.
      ``('type', 'SST_ASTRI_CHEC', 4.0)`` to apply to all telescopes of that type,
      or use a wildcard like "LST*", or "*" to set a pure default value for all
      telescopes.
    - ``'id'``:  argument is a specific telescope ID ``['id', 89, 5.0]``)

    These are evaluated in-order, so you can first set a default value, and then set
    values for specific telescopes or types to override them.

    Examples
    --------

    .. code-block:: python

        tel_param = [
            ('type', '*', 5.0),                       # default for all
            ('type', 'LST_*', 5.2),
            ('type', 'MST_MST_NectarCam', 4.0),
            ('type', 'MST_MST_FlashCam', 4.5),
            ('id', 34, 4.0),                   # override telescope 34 specifically
        ]

    .. code-block:: python

        tel_param = 4.0  # sets this value for all telescopes

    """

    klass = TelescopePatternList
    _valid_defaults = (object,)  # allow everything, we validate the default ourselves

    def __init__(self, trait, default_value=Undefined, **kwargs):
        """
        Create a new TelescopeParameter
        """
        self._help = ""

        if not isinstance(trait, TraitType):
            raise TypeError("trait must be a TraitType instance")

        self._trait = trait
        self.allow_none = kwargs.get("allow_none", False)
        if default_value != Undefined:
            default_value = self.validate(self, default_value)

        if "help" not in kwargs:
            self.help = "A TelescopeParameter"

        super().__init__(default_value=default_value, **kwargs)

    @property
    def help(self):
        sep = "." if not self._help.endswith(".") else ""
        return f"{self._help}{sep} {self._trait.help}"

    @help.setter
    def help(self, value):
        self._help = value

    def from_string(self, s):
        val = super().from_string(s)
        # for strings, parsing fails and traitlets returns None
        if val == [("type", "*", None)] and s != "None":
            val = [("type", "*", self._trait.from_string(s))]
        return val

    def _validate_entry(self, obj, value):
        if value is None and self.allow_none is True:
            return None
        return self._trait.validate(obj, value)

    def validate(self, obj, value):
        # Support a single value for all (check and convert into a default value)
        if not isinstance(value, list | List | UserList | TelescopePatternList):
            value = [("type", "*", self._validate_entry(obj, value))]

        # Check each value of list
        normalized_value = TelescopePatternList()

        for pattern in value:
            # now check for the standard 3-tuple of (command, argument, value)
            if len(pattern) != 3:
                raise TraitError(
                    "pattern should be a tuple of (command, argument, value)"
                )

            command, arg, val = pattern
            val = self._validate_entry(obj, val)

            if not isinstance(command, str):
                raise TraitError("command must be a string")

            if command not in ["type", "id"]:
                raise TraitError("command must be one of: 'type', 'id'")

            if command == "type":
                if not isinstance(arg, str):
                    raise TraitError("'type' argument should be a string")

            if command == "id":
                try:
                    arg = int(arg)
                except ValueError:
                    raise TraitError(f"Argument of 'id' should be an int (got '{arg}')")

            val = self._validate_entry(obj, val)
            normalized_value.append((command, arg, val))

        normalized_value._lookup = TelescopeParameterLookup(normalized_value)
        if isinstance(value, TelescopePatternList) and value._subarray is not None:
            normalized_value.attach_subarray(value._subarray)

        return normalized_value

    def set(self, obj, value):
        # Support a single value for all (check and convert into a default value)
        if not isinstance(value, list | List | UserList | TelescopePatternList):
            value = [("type", "*", self._validate_entry(obj, value))]

        # Retain existing subarray description
        # when setting new value for TelescopeParameter
        try:
            old_value = obj._trait_values[self.name]
        except KeyError:
            old_value = self.default_value

        super().set(obj, value)

        if getattr(old_value, "_subarray", None) is not None:
            obj._trait_values[self.name].attach_subarray(old_value._subarray)
