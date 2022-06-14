"""
Class and related functions to read DL1 (a,b) and/or DL2 (a) data
from an HDF5 file produced with ctapipe-process.
"""

import re
from collections import defaultdict
from typing import Dict

import numpy as np
from astropy.table import join, vstack, Table
import tables

from ..core import Component, traits, Provenance
from ..instrument import SubarrayDescription
from .astropy_helpers import read_table, join_allow_empty

__all__ = ["TableLoader"]

PARAMETERS_GROUP = "/dl1/event/telescope/parameters"
IMAGES_GROUP = "/dl1/event/telescope/images"
TRIGGER_TABLE = "/dl1/event/subarray/trigger"
SHOWER_TABLE = "/simulation/event/subarray/shower"
TRUE_IMAGES_GROUP = "/simulation/event/telescope/images"
TRUE_PARAMETERS_GROUP = "/simulation/event/telescope/parameters"

DL2_SUBARRAY_GROUP = "/dl2/event/subarray"
DL2_TELESCOPE_GROUP = "/dl2/event/telescope"

by_id_RE = re.compile(r"tel_\d+")


SUBARRAY_EVENT_KEYS = ["obs_id", "event_id"]
TELESCOPE_EVENT_KEYS = ["obs_id", "event_id", "tel_id"]


def _empty_telescope_events_table():
    """
    Create a new astropy table with correct column names and dtypes
    for telescope event based data.
    """
    return Table(names=TELESCOPE_EVENT_KEYS, dtype=[np.int64, np.int64, np.int16])


def _empty_subarray_events_table():
    """
    Create a new astropy table with correct column names and dtypes
    for subarray event based data.
    """
    return Table(names=SUBARRAY_EVENT_KEYS, dtype=[np.int64, np.int64])


def _get_structure(h5file):
    """
    Check if data is stored by telescope id or by telescope type in h5file.
    """

    if PARAMETERS_GROUP in h5file:
        group = h5file.root[PARAMETERS_GROUP]
    elif IMAGES_GROUP in h5file:
        group = h5file.root[IMAGES_GROUP]
    else:
        raise ValueError(f"No DL1 parameters data in h5file: {h5file}")

    key = next(iter(group._v_children))  # pylint: disable=protected-access

    if by_id_RE.fullmatch(key):
        return "by_id"

    return "by_type"


def _add_column_prefix(table, prefix, ignore=()):
    """
    Add a prefix to all columns in table besides columns in ``ignore``.
    """
    for col in set(table.colnames) - set(ignore):
        table.rename_column(col, f"{prefix}_{col}")


def _join_subarray_events(table1, table2):
    """Outer join two tables on the telescope subarray keys"""
    return join_allow_empty(table1, table2, SUBARRAY_EVENT_KEYS, "outer")


def _join_telescope_events(table1, table2):
    """Outer join two tables on the telescope event keys"""
    return join_allow_empty(table1, table2, TELESCOPE_EVENT_KEYS, "outer")


class TableLoader(Component):
    """
    Load telescope-event or subarray-event data from ctapipe HDF5 files

    This class provides high-level access to data stored in ctapipe HDF5 files,
    such as created by the ctapipe-process tool (`~ctapipe.tools.process.ProcessorTool`).

    The following `TableLoader` methods load data from all relevant tables,
    depending on the options, and joins them into single tables:

    * `TableLoader.read_subarray_events`
    * `TableLoader.read_telescope_events`
    * `TableLoader.read_telescope_events_by_type` retuns a dict with a table per
      telescope type, which is needed for e.g. DL1 image data that might have
      different shapes for each of the telescope types as tables do not support
      variable length columns.

    Attributes
    ----------
    subarray : `~ctapipe.instrument.SubarrayDescription`
        The subarray as read from `input_url`.
    """

    input_url = traits.Path(directory_ok=False, exists=True).tag(config=True)

    load_dl1_images = traits.Bool(False, help="load extracted images").tag(config=True)
    load_dl1_parameters = traits.Bool(
        True, help="load reconstructed image parameters"
    ).tag(config=True)

    load_dl2 = traits.Bool(False, help="load available dl2 stereo parameters").tag(
        config=True
    )

    load_simulated = traits.Bool(False, help="load simulated shower information").tag(
        config=True
    )
    load_true_images = traits.Bool(False, help="load simulated shower images").tag(
        config=True
    )
    load_true_parameters = traits.Bool(
        False, help="load image parameters obtained from true images"
    ).tag(config=True)
    load_trigger = traits.Bool(True, help="load subarray trigger information").tag(
        config=True
    )
    load_instrument = traits.Bool(
        False, help="join subarray instrument information to each event"
    ).tag(config=True)

    def __init__(self, input_url=None, **kwargs):
        # enable using input_url as posarg
        if input_url not in {None, traits.Undefined}:
            kwargs["input_url"] = input_url
        super().__init__(**kwargs)

        self.subarray = SubarrayDescription.from_hdf(self.input_url)
        self.h5file = tables.open_file(self.input_url, mode="r")

        Provenance().add_input_file(self.input_url, role="Event data")

        try:
            self.structure = _get_structure(self.h5file)
        except ValueError:
            self.structure = None

        self.instrument_table = None
        if self.load_instrument:
            table = self.subarray.to_table()
            optics = self.subarray.to_table(kind="optics")
            optics["optics_index"] = np.arange(len(optics))
            optics.remove_columns(["name", "description", "type"])
            table = join(
                table,
                optics,
                keys="optics_index",
                # conflicts for TAB_VER, TAB_TYPE, not needed here, ignore
                metadata_conflicts="silent",
            )

            table.remove_columns(["optics_index", "camera_index"])
            self.instrument_table = table

    def close(self):
        """Close the underlying hdf5 file"""
        self.h5file.close()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _read_telescope_table(self, group, tel_id):
        if self.structure == "by_id":
            key = f"{group}/tel_{tel_id:03d}"
            condition = None
        else:
            key = f"{group}/{self.subarray.tel[tel_id]!s}"
            condition = f"tel_id == {tel_id}"

        if key in self.h5file:
            table = read_table(self.h5file, key, condition=condition)
        else:
            table = _empty_telescope_events_table()

        return table

    def read_subarray_events(self):
        """Read subarray-based event information.

        Returns
        -------
        table: astropy.io.Table
            Table with primary index columns "obs_id" and "event_id".
        """
        table = _empty_subarray_events_table()

        if self.load_trigger:
            trigger = read_table(self.h5file, TRIGGER_TABLE)
            table = _join_subarray_events(table, trigger)

        if self.load_simulated and SHOWER_TABLE in self.h5file:
            showers = read_table(self.h5file, SHOWER_TABLE)
            table = _join_subarray_events(table, showers)

        if self.load_dl2:
            if DL2_SUBARRAY_GROUP in self.h5file:
                for group_name in self.h5file.root[DL2_SUBARRAY_GROUP]._v_children:
                    group_path = f"{DL2_SUBARRAY_GROUP}/{group_name}"
                    group = self.h5file.root[group_path]

                    for algorithm in group._v_children:
                        dl2 = read_table(self.h5file, f"{group_path}/{algorithm}")

                        # add the algorithm as prefix to distinguish multiple
                        # algorithms predicting the same quantities
                        _add_column_prefix(
                            dl2, prefix=algorithm, ignore=SUBARRAY_EVENT_KEYS
                        )
                        table = _join_subarray_events(table, dl2)
        return table

    def _read_telescope_events_for_id(self, tel_id):
        """Read telescope-based event information for a single telescope.

        This is the most low-level function doing the actual reading.

        Parameters
        ----------
        tel_id: int
            Telescope identification number.

        Returns
        -------
        table: astropy.io.Table
            Table with primary index columns "obs_id", "event_id" and "tel_id".
        """

        if tel_id is None:
            raise ValueError("Please, specify a telescope ID.")

        table = _empty_telescope_events_table()

        if self.load_dl1_parameters:
            parameters = self._read_telescope_table(PARAMETERS_GROUP, tel_id)
            table = _join_telescope_events(table, parameters)

        if self.load_dl1_images:
            images = self._read_telescope_table(IMAGES_GROUP, tel_id)
            table = _join_telescope_events(table, images)

        if self.load_dl2:
            if DL2_TELESCOPE_GROUP in self.h5file:
                for group_name in self.h5file[DL2_TELESCOPE_GROUP]._v_children:
                    group_path = f"{DL2_TELESCOPE_GROUP}/{group_name}"
                    group = self.h5file.root[group_path]

                    for algorithm in group._v_children:
                        path = f"{group_path}/{algorithm}"
                        dl2 = self._read_telescope_table(path, tel_id)

                        # add the algorithm as prefix to distinguish multiple
                        # algorithms predicting the same quantities
                        _add_column_prefix(
                            dl2, prefix=algorithm, ignore=TELESCOPE_EVENT_KEYS
                        )
                        table = _join_telescope_events(table, dl2)

        if self.load_true_images:
            true_images = self._read_telescope_table(TRUE_IMAGES_GROUP, tel_id)
            table = _join_telescope_events(table, true_images)

        if self.load_true_parameters:
            true_parameters = self._read_telescope_table(TRUE_PARAMETERS_GROUP, tel_id)

            for col in set(true_parameters.colnames) - set(TELESCOPE_EVENT_KEYS):
                true_parameters.rename_column(col, f"true_{col}")

            table = _join_telescope_events(table, true_parameters)

        if self.load_instrument:
            table = join_allow_empty(
                table, self.instrument_table, keys=["tel_id"], join_type="left"
            )

        return table

    def _read_telescope_events_for_ids(self, tel_ids):

        table = vstack(
            [self._read_telescope_events_for_id(tel_id) for tel_id in tel_ids]
        )

        return table

    def _join_subarray_info(self, table):
        subarray_events = self.read_subarray_events()
        table = join_allow_empty(
            table, subarray_events, keys=SUBARRAY_EVENT_KEYS, join_type="left"
        )
        return table

    def read_telescope_events(self, telescopes=None):
        """
        Read telescope-based event information.

        If the corresponding traitlets are True, also subarray event information
        is joined onto the table.

        Parameters
        ----------
        telescopes: Optional[List[Union[int, str, TelescopeDescription]]]
            A list containing any combination of telescope IDs and/or
            telescope descriptions. If None, all available telescopes are read.

        Returns
        -------
        events: astropy.io.Table
            Table with primary index columns "obs_id", "event_id" and "tel_id".
        """

        if telescopes is None:
            tel_ids = self.subarray.tel.keys()
        else:
            tel_ids = self.subarray.get_tel_ids(telescopes)

        table = self._read_telescope_events_for_ids(tel_ids)

        if any([self.load_trigger, self.load_simulated, self.load_dl2]):
            table = self._join_subarray_info(table)

        return table

    def read_telescope_events_by_type(self, telescopes=None) -> Dict[str, Table]:
        """Read telescope-based event information.

        Parameters
        ----------
        telescopes: List[Union[int, str, TelescopeDescription]]
            Any list containing a combination of telescope IDs or telescope_descriptions.

        Returns
        -------
        tables: dict(astropy.io.Table)
            Dictionary of tables organized by telescope types
            Table with primary index columns "obs_id", "event_id" and "tel_id".
        """

        if telescopes is None:
            tel_ids = self.subarray.tel.keys()
        else:
            tel_ids = self.subarray.get_tel_ids(telescopes)

        by_type = defaultdict(list)

        for tel_id in tel_ids:
            key = str(self.subarray.tel[tel_id])
            by_type[key].append(self._read_telescope_events_for_id(tel_id))

        by_type = {k: vstack(ts) for k, ts in by_type.items()}

        if any([self.load_trigger, self.load_simulated, self.load_dl2]):
            for key, table in by_type.items():
                by_type[key] = self._join_subarray_info(table)

        return by_type
