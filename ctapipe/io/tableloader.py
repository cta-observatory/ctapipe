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
GEOMETRY_GROUP = "/dl2/event/subarray/geometry"
TRIGGER_TABLE = "/dl1/event/subarray/trigger"
SHOWER_TABLE = "/simulation/event/subarray/shower"
TRUE_IMAGES_GROUP = "/simulation/event/telescope/images"
TRUE_PARAMETERS_GROUP = "/simulation/event/telescope/parameters"

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


class TableLoader(Component):
    """
    Load telescope-event or subarray-event data from ctapipe HDF5 files

    This class provides high-level access to data stored in ctapipe HDF5 files,
    such as created by the ctapipe-process tool (`~ctapipe.tools.process.ProcessorTool`).

    The following `TableLoader` methods load data from all relevant tables,
    depending on the options, and joins them into single tables:
    * `TableLoader.read_subarray_events`
    * `TableLoader.read_telescope_events`

    `TableLoader.read_telescope_events_by_type` retuns a dict with a table per
    telescope type, which is needed for e.g. DL1 image data that might have
    different shapes for each of the telescope types as tables do not support
    variable length columns.
    """

    input_url = traits.Path(directory_ok=False, exists=True).tag(config=True)

    load_dl1_images = traits.Bool(False, help="load extracted images").tag(config=True)
    load_dl1_parameters = traits.Bool(
        True, help="load reconstructed image parameters"
    ).tag(config=True)
    load_dl2_geometry = traits.Bool(
        False, help="load reconstructed shower geometry information"
    ).tag(config=True)
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
            table = join_allow_empty(table, trigger, SUBARRAY_EVENT_KEYS, "outer")

        if self.load_simulated and SHOWER_TABLE in self.h5file:
            showers = read_table(self.h5file, SHOWER_TABLE)
            table = join_allow_empty(table, showers, SUBARRAY_EVENT_KEYS, "outer")

        if self.load_dl2_geometry:
            shower_geometry_group = self.h5file.root[GEOMETRY_GROUP]

            for reconstructor in shower_geometry_group._v_children:
                geometry = read_table(self.h5file, f"{GEOMETRY_GROUP}/{reconstructor}")

                # rename DL2 columns to explicit reconstructor
                # TBD: we could skip this if only 1 reconstructor is present
                # or simply find another way to deal with multiple reconstructions
                for col in set(geometry.colnames) - set(SUBARRAY_EVENT_KEYS):
                    geometry.rename_column(col, f"{reconstructor}_{col}")

                table = join_allow_empty(table, geometry, SUBARRAY_EVENT_KEYS, "outer")

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
            table = join_allow_empty(
                table, parameters, join_type="outer", keys=TELESCOPE_EVENT_KEYS
            )

        if self.load_dl1_images:
            images = self._read_telescope_table(IMAGES_GROUP, tel_id)
            table = join_allow_empty(
                table, images, join_type="outer", keys=TELESCOPE_EVENT_KEYS
            )

        if self.load_true_images:
            true_images = self._read_telescope_table(TRUE_IMAGES_GROUP, tel_id)
            table = join_allow_empty(
                table, true_images, join_type="outer", keys=TELESCOPE_EVENT_KEYS
            )

        if self.load_true_parameters:
            true_parameters = self._read_telescope_table(TRUE_PARAMETERS_GROUP, tel_id)

            for col in set(true_parameters.colnames) - set(TELESCOPE_EVENT_KEYS):
                true_parameters.rename_column(col, f"true_{col}")

            table = join_allow_empty(
                table, true_parameters, join_type="outer", keys=TELESCOPE_EVENT_KEYS
            )

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

        if any([self.load_trigger, self.load_simulated, self.load_dl2_geometry]):
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

        if any([self.load_trigger, self.load_simulated, self.load_dl2_geometry]):
            for key, table in by_type.items():
                by_type[key] = self._join_subarray_info(table)

        return by_type
