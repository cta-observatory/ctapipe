import re

import tables
from astropy.table import join, vstack, Table

from ..core import Component, traits
from ..instrument import SubarrayDescription
from .astropy_helpers import read_table

PARAMETERS_GROUP = "/dl1/event/telescope/parameters"
IMAGES_GROUP = "/dl1/event/telescope/images"
TRIGGER_TABLE = "/dl1/event/subarray/trigger"

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

    def close(self):
        self.h5file.close()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def load_telescope_events(self, tel_id):
        table = None

        if self.load_dl1_parameters:
            if self.structure == "by_id":
                key = f"{PARAMETERS_GROUP}/tel_{tel_id:03d}"
                condition = None
            else:
                key = f"{PARAMETERS_GROUP}/{self.subarray.tel[tel_id]:s}"
                condition = f"tel_id == {tel_id}"

            if key in self.h5file:
                table = read_table(self.h5file, key, condition=condition)
            else:
                table = Table()

        if self.load_dl1_images:
            if self.structure == "by_id":
                key = f"{IMAGES_GROUP}/tel_{tel_id:03d}"
                condition = None
            else:
                key = f"{IMAGES_GROUP}/{self.subarray.tel[tel_id]:s}"
                condition = f"tel_id == {tel_id}"

            images = read_table(self.h5file, key, condition=condition)
            if table is None:
                table = images
            else:
                table = join(
                    table,
                    images,
                    keys=["obs_id", "event_id", "tel_id"],
                    join_type="outer",
                )

        if self.load_trigger:
            table = join(
                table,
                self.trigger_table,
                keys=["obs_id", "event_id"],
                join_type="inner",
            )

        return table
