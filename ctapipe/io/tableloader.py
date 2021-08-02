import re
import numpy as np

import tables
from astropy.table import join, vstack, Table

from ..core import Component, traits
from ..instrument import SubarrayDescription
from .astropy_helpers import read_table

__all__ = ["get_structure", "TableLoader"]

PARAMETERS_GROUP = "/dl1/event/telescope/parameters"
IMAGES_GROUP = "/dl1/event/telescope/images"
TRIGGER_TABLE = "/dl1/event/subarray/trigger"
SHOWER_TABLE = "/simulation/event/subarray/shower"
TRUE_IMAGES_GROUP = "/simulation/event/telescope/images"

by_id_RE = re.compile(r"tel_\d+")



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
    """ A helper class to load and join tables from a DL1 file"""

    input_url = traits.Path(directory_ok=False, exists=True).tag(config=True)

    load_dl1_images = traits.Bool(False).tag(config=True)
    load_dl1_parameters = traits.Bool(True).tag(config=True)
    load_dl2 = traits.Bool(False).tag(config=True)
    load_simulated = traits.Bool(False).tag(config=True)
    load_true_images = traits.Bool(False).tag(config=True)
    load_trigger = traits.Bool(True).tag(config=True)
    load_instrument = traits.Bool(False).tag(config=True)

    def __init__(self, input_url=None, **kwargs):
        # enable using input_url as posarg
        if input_url not in {None, traits.Undefined}:
            kwargs["input_url"] = input_url
        super().__init__(**kwargs)

        self.subarray = SubarrayDescription.from_hdf(self.input_url)
        self.h5file = tables.open_file(self.input_url, mode="r")

        try:
            self.structure = get_structure(self.h5file)
        except ValueError:
            self.structure = None

        self.trigger_table = read_table(self.h5file, TRIGGER_TABLE)

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

    def read_telescope_events(self, tel_ids=None):
        if tel_ids is None:
            tel_ids = self.subarray.tel.keys()

        return vstack([self.read_telescope_events_for_id(tel_id) for tel_id in tel_ids])

    def read_telescope_events_for_type(self, tel_type):
        tel_ids = self.subarray.get_tel_ids_for_type(tel_type)
        return self.read_telescope_events(tel_ids)

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

    def read_telescope_events_for_id(self, tel_id):
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

        if self.shower_table is not None:
            table = join(
                table, self.shower_table, keys=["obs_id", "event_id"], join_type="inner"
            )

        if self.load_trigger and len(table) > 0:
            table = join(
                table,
                self.trigger_table,
                keys=["obs_id", "event_id"],
                join_type="inner",
            )

        if self.load_instrument and len(table) > 0:
            table = join(
                table, self.instrument_table, keys=["tel_id"], join_type="inner"
            )

        return table
