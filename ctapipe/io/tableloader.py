"""
Class and functions to read DL1 (a,b) and/or DL2 (a) data from an HDF5 file produced with ctapipe-process.
"""

import re
from typing import List, Union

import numpy as np
import tables
from astropy.table import join, vstack, Table

from ..core import Component, traits, Provenance
from ..instrument import SubarrayDescription, TelescopeDescription
from .astropy_helpers import read_table

__all__ = ["get_tel_ids", "get_structure", "TableLoader"]

PARAMETERS_GROUP = "/dl1/event/telescope/parameters"
IMAGES_GROUP = "/dl1/event/telescope/images"
GEOMETRY_GROUP = "/dl2/event/subarray/geometry"
TRIGGER_TABLE = "/dl1/event/subarray/trigger"
SHOWER_TABLE = "/simulation/event/subarray/shower"
TRUE_IMAGES_GROUP = "/simulation/event/telescope/images"

by_id_RE = re.compile(r"tel_\d+")


def get_tel_ids(
    subarray: SubarrayDescription,
    telescopes: List[Union[int, str, TelescopeDescription]]
) -> List[int]:
    ids = set()

    for telescope in telescopes:
        if isinstance(telescope, int):
            ids.add(telescope)
        ids.update(subarray.get_tel_ids_for_type(telescope))

    return sorted(ids)


def get_structure(h5file):

    if PARAMETERS_GROUP in h5file:
        g = h5file.root[PARAMETERS_GROUP]
    elif IMAGES_GROUP in h5file:
        g = h5file.root[IMAGES_GROUP]
    else:
        raise ValueError(f"No DL1 parameters data in h5file: {h5file}")

    key = next(iter(g._v_children))
    if by_id_RE.fullmatch(key):
        return "by_id"
    else:
        return "by_type"


class TableLoader(Component):
    """ A helper class to load and join tables from an HDF5 file produced with ctapipe-process.

    It is recommended to use the methods 'read_events' for a single table based on telescope IDs and
    'read_events_by_type' for a dictionary of tables based on telescope types.
    """

    input_url = traits.Path(directory_ok=False, exists=True).tag(config=True)

    load_dl1_images = traits.Bool(False, help="load extracted images").tag(config=True)
    load_dl1_parameters = traits.Bool(
        True, help="load reconstructed image parameters").tag(config=True)
    load_dl2_geometry = traits.Bool(
        False, help="load reconstructed shower geometry information").tag(config=True)
    load_simulated = traits.Bool(
        False, help="load simulated shower information").tag(config=True)
    load_true_images = traits.Bool(
        False, help="load simulated shower images").tag(config=True)
    load_trigger = traits.Bool(
        True, help="load subarray trigger information").tag(config=True)
    load_instrument = traits.Bool(
        False, help="join subarray instrument information to each event").tag(config=True)

    def __init__(self, input_url=None, **kwargs):
        # enable using input_url as posarg
        if input_url not in {None, traits.Undefined}:
            kwargs["input_url"] = input_url
        super().__init__(**kwargs)

        self.subarray = SubarrayDescription.from_hdf(self.input_url)
        self.h5file = tables.open_file(self.input_url, mode="r")

        Provenance().add_input_file(self.input_url, role="Event data")

        try:
            self.structure = get_structure(self.h5file)
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

        self.shower_table = None
        if self.load_simulated and SHOWER_TABLE in self.h5file:
            self.shower_table = read_table(self.h5file, SHOWER_TABLE)

    def close(self):
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
            table = Table()

        return table

    def read_subarray_events(self, table=None):

        if self.shower_table:
            if (table is not None) and len(table) > 0:
                table = join(
                    table,
                    self.shower_table,
                    keys=["obs_id", "event_id"],
                    join_type="inner"
                )
            else:
                table = self.shower_table

        if self.load_trigger:
            if (table is not None) and len(table) > 0:
                table = join(
                    table,
                    read_table(self.h5file, TRIGGER_TABLE),
                    keys=["obs_id", "event_id"],
                    join_type="inner",
                )
            else:
                table = read_table(self.h5file, TRIGGER_TABLE)

        if self.load_dl2_geometry:

            shower_geometry_group = self.h5file.root[GEOMETRY_GROUP]

            for i, reconstructor in enumerate(shower_geometry_group._v_children):

                geometry = read_table(self.h5file, f"{GEOMETRY_GROUP}/{reconstructor}")

                # rename DL2 columns to explicit reconstructor
                # TBD: we could skip this if only 1 reconstructor is present
                # or simply find another way to deal with multiple reconstructions
                for col in set(geometry.colnames) - {"obs_id", "event_id"}:
                    geometry.rename_column(col, f"{reconstructor}_{col}")

                if ((table is None) or (len(table) == 0)) and (i == 0):
                    table = geometry
                else:
                    table = join(
                        table, geometry, keys=["obs_id", "event_id"], join_type="left"
                    )
        return table

    def read_telescope_events(self, tel_ids):

        table = vstack([self.read_telescope_events_for_id(tel_id) for tel_id in tel_ids])

        return table

    def read_events(self, labels=None):
        """Read telescope-based event information.

        Parameters
        ----------
        labels: list
            Any list combination of tel_ids, tel_types, or telescope_descriptions.

        Returns
        -------
        table: astropy.io.Table
            Table with primary columns "obs_id", "event_id" and "tel_id".
        """

        if labels is None:
            tel_ids = self.subarray.tel.keys()
        else:
            tel_ids = get_tel_ids(self.subarray, labels)

        if any([self.load_dl1_images, self.load_dl1_parameters, self.load_true_images]):
            table = self.read_telescope_events(tel_ids)
        else:
            table = None

        table = self.read_subarray_events(table)

        return table

    def read_events_by_tel_type(self, labels=None):
        """Read telescope-based event information.

        Parameters
        ----------
        labels: list
            Any list combination of tel_ids, tel_types, or telescope_descriptions.

        Returns
        -------
        tables: dict(astropy.io.Table)
            Dictionary of tables organized by telescope types
            with primary columns "obs_id", "event_id".
        """

        if labels is None:
            tel_ids = self.subarray.tel.keys()
        else:
            tel_ids = get_tel_ids(self.subarray, labels)

        selected_subarray = self.subarray.select_subarray(tel_ids)
        selected_tel_types = selected_subarray.telescope_types

        tables = {}
        for tel_type in selected_tel_types:

            tables[str(tel_type)] = self.read_events_for_type(tel_type)

        return tables

    def read_events_for_type(self, tel_type):
        """Read event information for a single telescope type.

        Parameters
        ----------
        tel_type: str or ctapipe.instrument.TelescopeDescription
            Identifier for the telescope describing type, optics and camera.

        Returns
        -------
        tables: dict(astropy.io.Table)
            Dictionary of tables organized by telescope types
            with primary columns "obs_id", "event_id".
        """

        if tel_type is None:
            raise ValueError("Please, specify a telescope description.")
        else:
            tel_ids = self.subarray.get_tel_ids_for_type(tel_type)

        if any([self.load_dl1_images, self.load_dl1_parameters, self.load_true_images]):
            table = self.read_telescope_events(tel_ids)
        else:
            table = None

        table = self.read_subarray_events(table)

        return table

    def read_telescope_events_for_id(self, tel_id):
        """Read telescope-based event information for a single telescope.

        Parameters
        ----------
        tel_id: int
            Telescope identification number.

        Returns
        -------
        table: astropy.io.Table
            Table with primary column "tel_id".
        """

        if tel_id is None:
            raise ValueError("Please, specify a telescope ID.")

        table = None

        if self.load_dl1_parameters:
            table = self._read_telescope_table(PARAMETERS_GROUP, tel_id)

        if self.load_dl1_images:
            images = self._read_telescope_table(IMAGES_GROUP, tel_id)

            if table is None or len(table) == 0:
                table = images
            else:
                table = join(
                    table,
                    images,
                    keys=["obs_id", "event_id", "tel_id"],
                    join_type="outer",
                )

        if self.load_true_images:
            true_images = self._read_telescope_table(TRUE_IMAGES_GROUP, tel_id)
            if table is None or len(table) == 0:
                table = true_images
            else:
                table = join(
                    table,
                    true_images,
                    keys=["obs_id", "event_id", "tel_id"],
                    join_type="outer",
                )

        if self.load_instrument and len(table) > 0:
            table = join(
                table, self.instrument_table, keys=["tel_id"], join_type="inner"
            )

        return table
