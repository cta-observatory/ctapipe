"""
Class and related functions to read DL1 (a,b) and/or DL2 (a) data
from an HDF5 file produced with ctapipe-process.
"""

from pathlib import Path
import re
from collections import defaultdict
from typing import Dict
from astropy.utils.decorators import lazyproperty

import numpy as np
from astropy.table import vstack, Table
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
TRUE_IMPACT_GROUP = "/simulation/event/telescope/impact"

DL2_SUBARRAY_GROUP = "/dl2/event/subarray"
DL2_TELESCOPE_GROUP = "/dl2/event/telescope"

by_id_RE = re.compile(r"tel_\d+")


SUBARRAY_EVENT_KEYS = ["obs_id", "event_id"]
TELESCOPE_EVENT_KEYS = ["obs_id", "event_id", "tel_id"]


class ChunkIterator:
    """An iterator that calls a function on advancemnt

    Parameters
    ----------
    func : Callable
        Signature must be ``function(*args, start=start, stop=stop, **kwargs)``
        Where start and stop are the first and last (noninclusive) indicies of
        a chunk.
    n_total : int
        Total number of elements
    chunk_size : size of one chunk, last chunk will have size <= chunk_size

    *args and **kwargs will be passed to `func`

    """

    def __init__(
        self,
        func,
        n_total,
        chunk_size,
        *args,
        **kwargs,
    ):
        self.func = func
        self.n_total = n_total
        self.chunk_size = chunk_size
        self._current_chunk = 0
        self.n_chunks = int(np.ceil(self.n_total / self.chunk_size))
        self.args = args
        self.kwargs = kwargs

    def __len__(self):
        return self.n_chunks

    def __iter__(self):
        self._current_chunk = 0
        return self

    def __next__(self):
        if self._current_chunk == self.n_chunks:
            raise StopIteration

        chunk = self._current_chunk
        start = chunk * self.chunk_size
        stop = min(self.n_total, (chunk + 1) * self.chunk_size)

        self._current_chunk += 1
        return self.func(*self.args, start=start, stop=stop, **self.kwargs)


def _empty_telescope_events_table():
    """
    Create a new astropy table with correct column names and dtypes
    for telescope event based data.
    """
    return Table(names=TELESCOPE_EVENT_KEYS, dtype=[np.int32, np.int64, np.int16])


def _empty_subarray_events_table():
    """
    Create a new astropy table with correct column names and dtypes
    for subarray event based data.
    """
    return Table(names=SUBARRAY_EVENT_KEYS, dtype=[np.int32, np.int64])


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

    input_url = traits.Path(
        directory_ok=False, exists=True, allow_none=True, default_value=None
    ).tag(config=True)

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

    def __init__(self, input_url=None, h5file=None, **kwargs):
        self._should_close = False

        # enable using input_url as posarg
        if input_url not in {None, traits.Undefined}:
            kwargs["input_url"] = input_url

        super().__init__(**kwargs)
        if h5file is None and self.input_url is None:
            raise ValueError("Need to specify either input_url or h5file")

        if h5file is None:
            self.h5file = tables.open_file(self.input_url, mode="r")
            self._should_close = True
        else:
            if not isinstance(h5file, tables.File):
                raise TypeError("h5file must be a tables.File")
            self.input_url = Path(h5file.filename)
            self.h5file = h5file

        self.subarray = SubarrayDescription.from_hdf(self.h5file)

        Provenance().add_input_file(self.input_url, role="Event data")

        try:
            self.structure = _get_structure(self.h5file)
        except ValueError:
            self.structure = None

        self.instrument_table = None
        if self.load_instrument:
            self.instrument_table = self.subarray.to_table("joined")

    def close(self):
        """Close the underlying hdf5 file"""
        if self._should_close:
            self.h5file.close()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _read_telescope_table(self, group, tel_id, start=None, stop=None):
        if self.structure == "by_id":
            key = f"{group}/tel_{tel_id:03d}"
            condition = None
        else:
            key = f"{group}/{self.subarray.tel[tel_id]!s}"
            condition = f"tel_id == {tel_id}"

        if key in self.h5file:
            table = read_table(
                self.h5file, key, condition=condition, start=start, stop=stop
            )
        else:
            table = _empty_telescope_events_table()

        return table

    def read_subarray_events(self, start=None, stop=None):
        """Read subarray-based event information.

        Returns
        -------
        table: astropy.io.Table
            Table with primary index columns "obs_id" and "event_id".
        """
        table = _empty_subarray_events_table()

        if self.load_trigger:
            trigger = read_table(self.h5file, TRIGGER_TABLE, start=start, stop=stop)
            table = _join_subarray_events(table, trigger)

        if self.load_simulated and SHOWER_TABLE in self.h5file:
            showers = read_table(self.h5file, SHOWER_TABLE, start=start, stop=stop)
            table = _join_subarray_events(table, showers)

        if self.load_dl2:
            if DL2_SUBARRAY_GROUP in self.h5file:
                for group_name in self.h5file.root[DL2_SUBARRAY_GROUP]._v_children:
                    group_path = f"{DL2_SUBARRAY_GROUP}/{group_name}"
                    group = self.h5file.root[group_path]

                    for algorithm in group._v_children:
                        dl2 = read_table(
                            self.h5file,
                            f"{group_path}/{algorithm}",
                            start=start,
                            stop=stop,
                        )

                        # add the algorithm as prefix to distinguish multiple
                        # algorithms predicting the same quantities
                        _add_column_prefix(
                            dl2, prefix=algorithm, ignore=SUBARRAY_EVENT_KEYS
                        )
                        table = _join_subarray_events(table, dl2)
        return table

    def read_subarray_events_chunked(self, chunk_size):
        """
        Iterate over chunks of subarray events.

        Parameters
        ----------
        chunk_size: int
            Number of subarray events to load per chunk
        """
        n_events = self.h5file.root[TRIGGER_TABLE].shape[0]

        return ChunkIterator(
            self.read_subarray_events,
            n_total=n_events,
            chunk_size=chunk_size,
        )

    def _read_telescope_events_for_id(self, tel_id, start=None, stop=None):
        """Read telescope-based event information for a single telescope.

        This is the most low-level function doing the actual reading.

        Parameters
        ----------
        tel_id: int
            Telescope identification number.
        start: int
            First row to read
        stop: int
            Last row to read (non-inclusive)

        Returns
        -------
        table: astropy.io.Table
            Table with primary index columns "obs_id", "event_id" and "tel_id".
        """
        if tel_id is None:
            raise ValueError("Please, specify a telescope ID.")

        table = _empty_telescope_events_table()

        if self.load_dl1_parameters:
            parameters = self._read_telescope_table(
                PARAMETERS_GROUP, tel_id, start=start, stop=stop
            )
            table = _join_telescope_events(table, parameters)

        if self.load_dl1_images:
            images = self._read_telescope_table(
                IMAGES_GROUP, tel_id, start=start, stop=stop
            )
            table = _join_telescope_events(table, images)

        if self.load_dl2:
            if DL2_TELESCOPE_GROUP in self.h5file:
                dl2_tel_group = self.h5file.root[DL2_TELESCOPE_GROUP]
                for group_name in dl2_tel_group._v_children:
                    group_path = f"{DL2_TELESCOPE_GROUP}/{group_name}"
                    group = self.h5file.root[group_path]

                    for algorithm in group._v_children:
                        path = f"{group_path}/{algorithm}"
                        dl2 = self._read_telescope_table(
                            path, tel_id, start=start, stop=stop
                        )

                        # add the algorithm as prefix to distinguish multiple
                        # algorithms predicting the same quantities
                        _add_column_prefix(
                            dl2, prefix=algorithm, ignore=TELESCOPE_EVENT_KEYS
                        )
                        table = _join_telescope_events(table, dl2)

        if self.load_true_images:
            true_images = self._read_telescope_table(
                TRUE_IMAGES_GROUP, tel_id, start=start, stop=stop
            )
            table = _join_telescope_events(table, true_images)

        if self.load_true_parameters:
            true_parameters = self._read_telescope_table(
                TRUE_PARAMETERS_GROUP, tel_id, start=start, stop=stop
            )
            _add_column_prefix(true_parameters, "true", ignore=TELESCOPE_EVENT_KEYS)
            table = _join_telescope_events(table, true_parameters)

        if self.load_instrument:
            table = join_allow_empty(
                table, self.instrument_table, keys=["tel_id"], join_type="left"
            )

        if self.load_simulated and TRUE_IMPACT_GROUP in self.h5file.root:
            impacts = self._read_telescope_table(
                TRUE_IMPACT_GROUP, tel_id, start=start, stop=stop
            )
            table = _join_telescope_events(table, impacts)

        return table

    def _read_telescope_events_for_ids(self, tel_ids, tel_start=None, tel_stop=None):
        if tel_start is None:
            tel_start = [None] * len(tel_ids)

        if tel_stop is None:
            tel_stop = [None] * len(tel_ids)

        table = vstack(
            [
                self._read_telescope_events_for_id(tel_id, start=start, stop=stop)
                for tel_id, start, stop in zip(tel_ids, tel_start, tel_stop)
            ]
        )

        return table

    def _join_subarray_info(self, table, start=None, stop=None):
        subarray_events = self.read_subarray_events(start=start, stop=stop)
        table = join_allow_empty(
            table, subarray_events, keys=SUBARRAY_EVENT_KEYS, join_type="left"
        )
        return table

    def read_telescope_events(self, telescopes=None, start=None, stop=None):
        """
        Read telescope-based event information.

        If the corresponding traitlets are True, also subarray event information
        is joined onto the table.

        The start, stop parameters enable to only load parts of the file,
        note however, to maintain integrity of subarray events, these
        refer to the subarray indices in the file. E.g. ``start=0``, ``stop=10``
        would load the telescope events corresponding to the first 10 subarray
        events in the input file.

        Parameters
        ----------
        telescopes: Optional[List[Union[int, str, TelescopeDescription]]]
            A list containing any combination of telescope IDs and/or
            telescope descriptions. If None, all available telescopes are read.
        start: int
            First *subarray* event to read
        stop: int
            Last *subarray* event (non-inclusive)

        Returns
        -------
        events: astropy.io.Table
            Table with primary index columns "obs_id", "event_id" and "tel_id".
        """

        if telescopes is None:
            tel_ids = tuple(self.subarray.tel.keys())
        else:
            tel_ids = self.subarray.get_tel_ids(telescopes)

        tel_start = None
        tel_stop = None
        if start is not None or stop is not None:

            indices = self.subarray.tel_ids_to_indices(tel_ids)

            # find first/last row for each telescope
            if start is not None:
                tel_start = self._n_telescope_events[start][indices]

            if stop is not None:
                if stop >= len(self._n_telescope_events):
                    tel_stop = None
                else:
                    tel_stop = self._n_telescope_events[stop][indices]

        table = self._read_telescope_events_for_ids(tel_ids, tel_start, tel_stop)

        if any([self.load_trigger, self.load_simulated, self.load_dl2]):
            table = self._join_subarray_info(table, start=start, stop=stop)

        return table

    def read_telescope_events_chunked(self, chunk_size, **kwargs):
        """
        Iterate over chunks of telescope events.

        Parameters
        ----------
        chunk_size: int
            Number of subarray events to load per chunk.
            The telescope tables might be larger or smaller than chunk_size
            depending on the selected telescopes.

        **kwargs are passed to `read_telescope_events`
        """
        n_events = self.h5file.root[TRIGGER_TABLE].shape[0]

        return ChunkIterator(
            self.read_telescope_events,
            n_total=n_events,
            chunk_size=chunk_size,
            **kwargs,
        )

    @lazyproperty
    def _n_telescope_events(self):
        """
        Number of telescope events in the file for each telescope previous
        to the nth subarray event.
        """
        # we need to load the trigger table until "stop" to
        # know which telescopes participated in which events
        table = self.h5file.root[TRIGGER_TABLE]
        n_events = table.shape[0]
        n_telescopes = table.coldescrs["tels_with_trigger"].shape[0]

        tels_with_trigger = np.zeros((n_events + 1, n_telescopes), dtype=np.bool_)
        table.read(field="tels_with_trigger", out=tels_with_trigger[1:])
        tels_with_trigger = tels_with_trigger.astype(np.uint32)

        np.cumsum(tels_with_trigger, out=tels_with_trigger, axis=0)
        return tels_with_trigger

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
