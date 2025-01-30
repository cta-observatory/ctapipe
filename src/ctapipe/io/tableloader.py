"""
Class and related functions to read DL1 (a,b) and/or DL2 (a) data
from an HDF5 file produced with ctapipe-process.
"""
import warnings
from collections import defaultdict, namedtuple
from pathlib import Path

import numpy as np
import tables
from astropy.table import Table, hstack, vstack
from astropy.utils.decorators import lazyproperty

from ..core import Component, Provenance, traits
from ..instrument import FocalLengthKind, SubarrayDescription
from ..monitoring.interpolation import PointingInterpolator
from .astropy_helpers import join_allow_empty, read_table

__all__ = ["TableLoader"]

PARAMETERS_GROUP = "/dl1/event/telescope/parameters"
IMAGES_GROUP = "/dl1/event/telescope/images"
MUON_GROUP = "/dl1/event/telescope/muon"
TRIGGER_TABLE = "/dl1/event/subarray/trigger"
SHOWER_TABLE = "/simulation/event/subarray/shower"
TRUE_IMAGES_GROUP = "/simulation/event/telescope/images"
TRUE_PARAMETERS_GROUP = "/simulation/event/telescope/parameters"
TRUE_IMPACT_GROUP = "/simulation/event/telescope/impact"
SIMULATION_CONFIG_TABLE = "/configuration/simulation/run"
SHOWER_DISTRIBUTION_TABLE = "/simulation/service/shower_distribution"
OBSERVATION_TABLE = "/configuration/observation/observation_block"
FIXED_POINTING_GROUP = "/configuration/telescope/pointing"
POINTING_GROUP = "/dl0/monitoring/telescope/pointing"

DL2_SUBARRAY_GROUP = "/dl2/event/subarray"
DL2_TELESCOPE_GROUP = "/dl2/event/telescope"

SUBARRAY_EVENT_KEYS = ["obs_id", "event_id"]
TELESCOPE_EVENT_KEYS = ["obs_id", "event_id", "tel_id"]


Chunk = namedtuple("Chunk", ["start", "stop", "data"])


class IndexNotMatching(UserWarning):
    """Warning that is raised if the order of two tables is not matching as expected"""


class ChunkIterator:
    """An iterator that calls a function on advancemnt

    Parameters
    ----------
    func : Callable
        Signature must be ``function(*args, start=start, stop=stop, **kwargs)``
        Where start and stop are the first and last (noninclusive) indices of
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
        args,
        kwargs,
    ):
        self.func = func
        self.n_total = n_total
        self.chunk_size = chunk_size
        self.n_chunks = int(np.ceil(self.n_total / self.chunk_size))
        self.args = args
        self.kwargs = kwargs

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, chunk):
        if chunk < 0:
            chunk = self.n_chunks - chunk

        if chunk >= self.n_chunks:
            raise IndexError(
                f"Index {chunk} is out of bounds for {self.__class__.__name__}"
                f" of length {len(self)}"
            )

        start = chunk * self.chunk_size
        stop = min(self.n_total, (chunk + 1) * self.chunk_size)
        return Chunk(
            start, stop, self.func(*self.args, start=start, stop=stop, **self.kwargs)
        )


def _empty_telescope_events_table():
    """
    Create a new astropy table with correct column names and dtypes
    for telescope event based data.
    """
    return Table(names=TELESCOPE_EVENT_KEYS, dtype=[np.uint64, np.uint64, np.uint16])


def _join_subarray_events(table1, table2):
    """Outer join two tables on the telescope subarray keys"""
    return join_allow_empty(table1, table2, SUBARRAY_EVENT_KEYS, "left")


def _join_telescope_events(table1, table2):
    """Outer join two tables on the telescope event keys"""
    # we start with an empty table, but after the first non-empty, we perform
    # left joins
    if len(table1) == 0:
        how = "right"
    else:
        how = "left"
    return join_allow_empty(table1, table2, TELESCOPE_EVENT_KEYS, how)


def _merge_table_same_index(table1, table2, index_keys, fallback_join_type="left"):
    """Merge two tables assuming their primary keys are identical"""
    if len(table1) != len(table2):
        raise ValueError(
            f"Tables must have identical length, {len(table1)} != {len(table2)}"
        )

    if len(table1) == 0:
        return table1

    if not np.all(table1[index_keys] == table2[index_keys]):
        warnings.warn(
            "Table order does not match, falling back to join", IndexNotMatching
        )
        return join_allow_empty(table1, table2, index_keys, fallback_join_type)

    columns = [col for col in table2.columns if col not in index_keys]
    return hstack((table1, table2[columns]), join_type="exact")


def _merge_subarray_tables(table1, table2):
    return _merge_table_same_index(table1, table2, SUBARRAY_EVENT_KEYS)


def _merge_telescope_tables(table1, table2):
    return _merge_table_same_index(table1, table2, TELESCOPE_EVENT_KEYS)


class TableLoader(Component):
    """
    Load telescope-event or subarray-event data from ctapipe HDF5 files

    This class provides high-level access to data stored in ctapipe HDF5 files,
    such as created by the ctapipe-process tool (`~ctapipe.tools.process.ProcessorTool`).

    The following `TableLoader` methods load data from all relevant tables,
    depending on the options, and joins them into single tables:

    * `TableLoader.read_subarray_events`
    * `TableLoader.read_telescope_events`
    * `TableLoader.read_telescope_events_by_type` returns a dict with a table per
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
    dl1_images = traits.Bool(False, help="load extracted images").tag(config=True)
    dl1_parameters = traits.Bool(True, help="load reconstructed image parameters").tag(
        config=True
    )
    dl1_muons = traits.Bool(False, help="load muon ring parameters").tag(config=True)

    dl2 = traits.Bool(True, help="load available dl2 stereo parameters").tag(
        config=True
    )

    simulated = traits.Bool(True, help="load simulated shower information").tag(
        config=True
    )
    true_images = traits.Bool(False, help="load simulated shower images").tag(
        config=True
    )
    true_parameters = traits.Bool(
        True, help="load image parameters obtained from true images"
    ).tag(config=True)

    instrument = traits.Bool(
        False, help="join subarray instrument information to each event"
    ).tag(config=True)

    observation_info = traits.Bool(
        False, help="join observation information to each event"
    ).tag(config=True)

    pointing = traits.Bool(
        True,
        help="Load pointing information and interpolate / join to events",
    ).tag(config=True)

    focal_length_choice = traits.UseEnum(
        FocalLengthKind,
        default_value=FocalLengthKind.EFFECTIVE,
        help=(
            "If both nominal and effective focal lengths are available, "
            " which one to use for the `~ctapipe.coordinates.CameraFrame` attached"
            " to the `~ctapipe.instrument.CameraGeometry` instances in the"
            " `ctapipe.instrument.SubarrayDescription`, which will be used in"
            " CameraFrame to TelescopeFrame coordinate transforms."
            " The 'nominal' focal length is the one used during "
            " the simulation, the 'effective' focal length is computed using specialized "
            " ray-tracing from a point light source"
        ),
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

        self.subarray = SubarrayDescription.from_hdf(
            self.h5file,
            focal_length_choice=self.focal_length_choice,
        )

        Provenance().add_input_file(self.input_url, role="Event data")

        self._pointing_interpolator = PointingInterpolator(
            h5file=self.h5file,
            parent=self,
        )

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

    def __len__(self):
        """Number of subarray events in input file"""
        return self.h5file.root[TRIGGER_TABLE].shape[0]

    def _check_args(self, **kwargs):
        """Checking args:
        1) If None, set to default (trait) value.
        2) If True but correlated group is not included in input file - set to False.
        returns a dict with new args"""
        groups = {
            "dl1_parameters": PARAMETERS_GROUP,
            "dl1_images": IMAGES_GROUP,
            "dl1_muons": MUON_GROUP,
            "true_parameters": TRUE_PARAMETERS_GROUP,
            "true_images": TRUE_IMAGES_GROUP,
            "observation_info": OBSERVATION_TABLE,
        }
        updated_attributes = {}
        for key, value in kwargs.items():
            updated_value = value
            if updated_value is None:
                updated_value = getattr(self, key)
            if updated_value and key in groups and groups[key] not in self.h5file.root:
                self.log.info(
                    "Setting %s to False, input file does not contain such data", key
                )
                updated_value = False
            updated_attributes[key] = updated_value
        return updated_attributes

    def _read_telescope_table(self, group, tel_id, start=None, stop=None):
        key = f"{group}/tel_{tel_id:03d}"

        if key in self.h5file:
            table = read_table(self.h5file, key, start=start, stop=stop)
        else:
            table = _empty_telescope_events_table()

        return table

    def _get_sort_index(self, start=None, stop=None):
        """
        Get plain index of increasing integers in the order in the file.

        Astropy.table.join orders by the join keys, we want to keep
        the original order. Adding an index before the first join and then
        using that in the end to sort back to the original order solves this.
        """
        table = read_table(
            self.h5file,
            TRIGGER_TABLE,
            start=start,
            stop=stop,
        )[["obs_id", "event_id"]]
        self._add_index_if_needed(table)
        return table

    @staticmethod
    def _sort_to_original_order(table, include_tel_id=False, keep_index=False):
        if len(table) == 0:
            return
        if include_tel_id:
            table.sort(("__index__", "tel_id"))
        else:
            table.sort("__index__")

        if not keep_index:
            table.remove_column("__index__")

    @staticmethod
    def _add_index_if_needed(table):
        if "__index__" not in table.colnames:
            table["__index__"] = np.arange(len(table))
            return True
        return False

    def read_simulation_configuration(self):
        """
        Read the simulation configuration table
        """
        return read_table(self.h5file, SIMULATION_CONFIG_TABLE)

    def read_shower_distribution(self):
        """
        Read the simulated shower distribution histograms
        """
        return read_table(self.h5file, SHOWER_DISTRIBUTION_TABLE)

    def read_observation_information(self):
        """
        Read the observation information
        """
        return read_table(self.h5file, OBSERVATION_TABLE)

    def _join_observation_info(self, table):
        obs_table = self.read_observation_information()
        # in v0.17, obs_id had inconsistent dtypes in different tables
        # Joining then gets messed up then because a join between int32 and uint64
        # casts the obs_id in the joint result to float.
        obs_table["obs_id"] = obs_table["obs_id"].astype(table["obs_id"].dtype)

        return join_allow_empty(
            table,
            obs_table,
            keys="obs_id",
            join_type="left",
            keep_order=True,
        )

    def read_subarray_events(
        self,
        start=None,
        stop=None,
        dl2=None,
        simulated=None,
        observation_info=None,
        keep_order=True,
    ):
        """Read subarray-based event information.

        Parameters
        ----------

        dl2: bool
            load available dl2 stereo parameters
        simulated: bool
            load simulated shower information
        observation_info: bool
            join observation information to each event
        start: int
            First *subarray* event to read
        stop: int
            Last *subarray* event (non-inclusive)

        Returns
        -------
        table: astropy.io.Table
            Table with primary index columns "obs_id" and "event_id".
        """
        updated_args = self._check_args(
            dl2=dl2,
            simulated=simulated,
            observation_info=observation_info,
        )
        dl2 = updated_args["dl2"]
        simulated = updated_args["simulated"]
        observation_info = updated_args["observation_info"]

        table = read_table(self.h5file, TRIGGER_TABLE, start=start, stop=stop)
        if keep_order:
            self._add_index_if_needed(table)

        if simulated and SHOWER_TABLE in self.h5file:
            showers = read_table(self.h5file, SHOWER_TABLE, start=start, stop=stop)
            table = _merge_subarray_tables(table, showers)

        if dl2:
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
                        table = _merge_subarray_tables(table, dl2)

        if observation_info:
            table = self._join_observation_info(table)

        if keep_order:
            self._sort_to_original_order(table)
        return table

    def read_subarray_events_chunked(self, chunk_size, *args, **kwargs):
        """
        Iterate over chunks of subarray events.

        Parameters
        ----------
        chunk_size: int
            Number of subarray events to load per chunk
        """
        return ChunkIterator(
            self.read_subarray_events,
            n_total=len(self),
            chunk_size=chunk_size,
            args=args,
            kwargs=kwargs,
        )

    def _read_telescope_events_for_id(
        self,
        tel_id,
        dl1_images,
        dl1_parameters,
        dl1_muons,
        dl2,
        simulated,
        true_images,
        true_parameters,
        instrument,
        pointing,
        start=None,
        stop=None,
    ):
        """Read telescope-based event information for a single telescope.

        This is the most low-level function doing the actual reading.

        Parameters
        ----------
        tel_id: int
            Telescope identification number.
        start: int
            First *subarray* event to read
        stop: int
            Last *subarray* event (non-inclusive)
        dl1_images: bool
            load extracted images
        dl1_parameters: bool
            load reconstructed image parameters
        dl1_muons: bool
            load muon ring parameters
        dl2: bool
            load available dl2 stereo parameters
        simulated: bool
            load simulated shower information
        true_images: bool
            load simulated shower images
        true_parameters: bool
            load image parameters obtained from true images
        instrument: bool
            join subarray instrument information to each event
        pointing: bool
            join pointing information to each event

        Returns
        -------
        table: astropy.io.Table
            Table with primary index columns "obs_id", "event_id" and "tel_id".
        """
        if tel_id is None:
            raise ValueError("Please, specify a telescope ID.")

        # trigger is stored in a single table for all telescopes, we need to
        # calculate the range to read from the stereo trigger info
        trigger_start = trigger_stop = None
        tel_start = tel_stop = None

        if start is not None or stop is not None:
            tel_start, tel_stop = self._get_tel_start_stop(tel_id, start, stop)

        if start is not None:
            trigger_start = self._n_total_telescope_events[start]

        if stop is not None:
            trigger_stop = self._n_total_telescope_events[stop]

        table = read_table(
            self.h5file,
            "/dl1/event/telescope/trigger",
            condition=f"tel_id == {tel_id}",
            start=trigger_start,
            stop=trigger_stop,
        )

        if dl1_parameters:
            parameters = self._read_telescope_table(
                PARAMETERS_GROUP, tel_id, start=tel_start, stop=tel_stop
            )
            table = _merge_telescope_tables(table, parameters)

        if dl1_muons:
            muon_parameters = self._read_telescope_table(
                MUON_GROUP, tel_id, start=tel_start, stop=tel_stop
            )
            table = _merge_telescope_tables(table, muon_parameters)

        if dl1_images:
            images = self._read_telescope_table(
                IMAGES_GROUP, tel_id, start=tel_start, stop=tel_stop
            )
            table = _merge_telescope_tables(table, images)

        if dl2:
            if DL2_TELESCOPE_GROUP in self.h5file:
                dl2_tel_group = self.h5file.root[DL2_TELESCOPE_GROUP]
                for group_name in dl2_tel_group._v_children:
                    group_path = f"{DL2_TELESCOPE_GROUP}/{group_name}"
                    group = self.h5file.root[group_path]

                    for algorithm in group._v_children:
                        path = f"{group_path}/{algorithm}"
                        dl2 = self._read_telescope_table(
                            path, tel_id, start=tel_start, stop=tel_stop
                        )
                        if len(dl2) == 0:
                            continue

                        table = _merge_telescope_tables(table, dl2)

        if true_images:
            true_images = self._read_telescope_table(
                TRUE_IMAGES_GROUP, tel_id, start=tel_start, stop=tel_stop
            )
            table = _merge_telescope_tables(table, true_images)

        if true_parameters:
            true_parameters = self._read_telescope_table(
                TRUE_PARAMETERS_GROUP, tel_id, start=tel_start, stop=tel_stop
            )
            table = _join_telescope_events(table, true_parameters)

        if instrument:
            instrument_table = self.subarray.to_table("joined")
            table = join_allow_empty(
                table, instrument_table, keys=["tel_id"], join_type="left"
            )

        if simulated and TRUE_IMPACT_GROUP in self.h5file.root:
            impacts = self._read_telescope_table(
                TRUE_IMPACT_GROUP,
                tel_id,
                start=tel_start,
                stop=tel_stop,
            )
            table = _join_telescope_events(table, impacts)

        if len(table) > 0 and pointing:
            # prefer monitoring pointing
            if POINTING_GROUP in self.h5file.root:
                alt, az = self._pointing_interpolator(tel_id, table["time"])
                table["telescope_pointing_altitude"] = alt
                table["telescope_pointing_azimuth"] = az
            elif FIXED_POINTING_GROUP in self.h5file.root:
                pointing_table = read_table(
                    self.h5file, f"{FIXED_POINTING_GROUP}/tel_{tel_id:03d}"
                )
                # we know that we only have the tel_id we are looking for, remove optimize joining
                del pointing_table["tel_id"]
                table = join_allow_empty(
                    table, pointing_table, ["obs_id"], "left", keep_order=True
                )

        return table

    def _read_telescope_events_for_ids(
        self,
        tel_ids,
        dl1_images,
        dl1_parameters,
        dl1_muons,
        dl2,
        simulated,
        true_images,
        true_parameters,
        instrument,
        pointing,
        start=None,
        stop=None,
    ):
        tables = [
            self._read_telescope_events_for_id(
                tel_id,
                dl1_images=dl1_images,
                dl1_parameters=dl1_parameters,
                dl1_muons=dl1_muons,
                dl2=dl2,
                simulated=simulated,
                true_images=true_images,
                true_parameters=true_parameters,
                instrument=instrument,
                pointing=pointing,
                start=start,
                stop=stop,
            )
            for tel_id in tel_ids
        ]
        return vstack(tables)

    def _join_subarray_info(
        self,
        table,
        dl2,
        simulated,
        observation_info,
        start=None,
        stop=None,
        subarray_events=None,
    ):
        if subarray_events is None:
            subarray_events = self.read_subarray_events(
                start=start,
                stop=stop,
                dl2=dl2,
                simulated=simulated,
                observation_info=observation_info,
                keep_order=False,
            )
        table = join_allow_empty(
            table,
            subarray_events,
            keys=SUBARRAY_EVENT_KEYS,
            join_type="left",
            # add suffix mono on duplicated columns, avoid underscore for stereo
            table_names=["_mono", ""],
            uniq_col_name="{col_name}{table_name}",
        )
        return table

    def _get_tel_start_stop(self, tel_id, start, stop):
        tel_start = None
        tel_stop = None
        index = self.subarray.tel_ids_to_indices(tel_id)[0]

        # find first/last row for each telescope
        if start is not None:
            tel_start = self._n_telescope_events[start, index]

        if stop is None or stop >= len(self._n_telescope_events):
            tel_stop = None
        else:
            tel_stop = self._n_telescope_events[stop, index]

        return tel_start, tel_stop

    def read_telescope_events(
        self,
        telescopes=None,
        start=None,
        stop=None,
        dl1_images=None,
        dl1_parameters=None,
        dl1_muons=None,
        dl2=None,
        simulated=None,
        true_images=None,
        true_parameters=None,
        instrument=None,
        observation_info=None,
        pointing=None,
    ):
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
        dl1_images: bool
            load extracted images
        dl1_parameters: bool
            load reconstructed image parameters
        dl1_muons: bool
            load muon ring parameters
        dl2: bool
            load available dl2 stereo parameters
        simulated: bool
            load simulated shower information
        true_images: bool
            load simulated shower images
        true_parameters: bool
            load image parameters obtained from true images
        instrument: bool
            join subarray instrument information to each event
        observation_info: bool
            join observation information to each event
        pointing: bool
            join pointing information to each event

        Returns
        -------
        events: astropy.io.Table
            Table with primary index columns "obs_id", "event_id" and "tel_id".
        """
        updated_args = self._check_args(
            dl1_images=dl1_images,
            dl1_parameters=dl1_parameters,
            dl1_muons=dl1_muons,
            dl2=dl2,
            simulated=simulated,
            true_images=true_images,
            true_parameters=true_parameters,
            instrument=instrument,
            observation_info=observation_info,
            pointing=pointing,
        )
        dl1_images = updated_args["dl1_images"]
        dl1_parameters = updated_args["dl1_parameters"]
        dl1_muons = updated_args["dl1_muons"]
        dl2 = updated_args["dl2"]
        simulated = updated_args["simulated"]
        true_images = updated_args["true_images"]
        true_parameters = updated_args["true_parameters"]
        instrument = updated_args["instrument"]
        observation_info = updated_args["observation_info"]
        pointing = updated_args["pointing"]

        if telescopes is None:
            tel_ids = tuple(self.subarray.tel.keys())
        else:
            tel_ids = self.subarray.get_tel_ids(telescopes)

        table = self._read_telescope_events_for_ids(
            tel_ids,
            dl1_images=dl1_images,
            dl1_parameters=dl1_parameters,
            dl1_muons=dl1_muons,
            dl2=dl2,
            simulated=simulated,
            true_images=true_images,
            true_parameters=true_parameters,
            instrument=instrument,
            pointing=pointing,
            start=start,
            stop=stop,
        )

        table = self._join_subarray_info(
            table,
            dl2=dl2,
            simulated=simulated,
            observation_info=observation_info,
            start=start,
            stop=stop,
        )

        # sort back to order in the file
        table = _join_subarray_events(
            table, self._get_sort_index(start=start, stop=stop)
        )
        self._sort_to_original_order(table, include_tel_id=True)

        return table

    def read_telescope_events_chunked(self, chunk_size, *args, **kwargs):
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
        return ChunkIterator(
            self.read_telescope_events,
            n_total=len(self),
            chunk_size=chunk_size,
            args=args,
            kwargs=kwargs,
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

    @lazyproperty
    def _n_total_telescope_events(self):
        """
        Number of telescope events in the file for each telescope previous
        to the nth subarray event.
        """
        return self._n_telescope_events.sum(axis=1)

    def read_telescope_events_by_type(
        self,
        telescopes=None,
        start=None,
        stop=None,
        dl1_images=None,
        dl1_parameters=None,
        dl1_muons=None,
        dl2=None,
        simulated=None,
        true_images=None,
        true_parameters=None,
        instrument=None,
        observation_info=None,
        pointing=None,
    ) -> dict[str, Table]:
        """Read subarray-based event information.

        Parameters
        ----------
        telescopes: List[Union[int, str, TelescopeDescription]]
            Any list containing a combination of telescope IDs or telescope_descriptions.
        dl1_images: bool
            load extracted images
        dl1_parameters: bool
            load reconstructed image parameters
        dl1_muons: bool
            load muon ring parameters
        dl2: bool
            load available dl2 stereo parameters
        simulated: bool
            load simulated shower information
        true_images: bool
            load simulated shower images
        true_parameters: bool
            load image parameters obtained from true images
        instrument: bool
            join subarray instrument information to each event
        observation_info: bool
            join observation information to each event
        pointing: bool
            join pointing information to each event

        Returns
        -------
        tables: dict(astropy.io.Table)
            Dictionary of tables organized by telescope types
            Table with primary index columns "obs_id", "event_id" and "tel_id".
        """
        updated_args = self._check_args(
            dl1_images=dl1_images,
            dl1_parameters=dl1_parameters,
            dl1_muons=dl1_muons,
            dl2=dl2,
            simulated=simulated,
            true_images=true_images,
            true_parameters=true_parameters,
            instrument=instrument,
            observation_info=observation_info,
            pointing=pointing,
        )
        dl1_images = updated_args["dl1_images"]
        dl1_parameters = updated_args["dl1_parameters"]
        dl1_muons = updated_args["dl1_muons"]
        dl2 = updated_args["dl2"]
        simulated = updated_args["simulated"]
        true_images = updated_args["true_images"]
        true_parameters = updated_args["true_parameters"]
        instrument = updated_args["instrument"]
        observation_info = updated_args["observation_info"]
        pointing = updated_args["pointing"]

        if telescopes is None:
            tel_ids = tuple(self.subarray.tel.keys())
        else:
            tel_ids = self.subarray.get_tel_ids(telescopes)

        subarray_events = self.read_subarray_events(
            start=start,
            stop=stop,
            dl2=dl2,
            simulated=simulated,
            observation_info=observation_info,
            keep_order=False,
        )
        self._add_index_if_needed(subarray_events)

        by_type = defaultdict(list)
        for tel_id in tel_ids:
            key = str(self.subarray.tel[tel_id])
            table = self._read_telescope_events_for_id(
                tel_id,
                start=start,
                stop=stop,
                dl1_images=dl1_images,
                dl1_parameters=dl1_parameters,
                dl1_muons=dl1_muons,
                dl2=dl2,
                simulated=simulated,
                true_images=true_images,
                true_parameters=true_parameters,
                instrument=instrument,
                pointing=pointing,
            )

            if len(table) > 0:
                by_type[key].append(table)

        by_type = {k: vstack(ts) for k, ts in by_type.items()}
        for key in by_type.keys():
            by_type[key] = self._join_subarray_info(
                by_type[key],
                dl2=dl2,
                simulated=simulated,
                observation_info=observation_info,
                subarray_events=subarray_events,
            )
            self._sort_to_original_order(by_type[key], include_tel_id=True)

        return by_type

    def read_telescope_events_by_type_chunked(self, chunk_size, *args, **kwargs):
        """
        Iterate over chunks of telescope events as dicts of telescope type to tables.

        Parameters
        ----------
        chunk_size: int
            Number of subarray events to load per chunk.
            The telescope tables might be larger or smaller than chunk_size
            depending on the selected telescopes.

        **kwargs are passed to `read_telescope_events`
        """
        return ChunkIterator(
            self.read_telescope_events_by_type,
            n_total=len(self),
            chunk_size=chunk_size,
            args=args,
            kwargs=kwargs,
        )

    def read_telescope_events_by_id(
        self,
        telescopes=None,
        start=None,
        stop=None,
        dl1_images=None,
        dl1_parameters=None,
        dl1_muons=None,
        dl2=None,
        simulated=None,
        true_images=None,
        true_parameters=None,
        instrument=None,
        observation_info=None,
        pointing=None,
    ) -> dict[int, Table]:
        """Read subarray-based event information.

        Parameters
        ----------
        telescopes: List[Union[int, str, TelescopeDescription]]
            Any list containing a combination of telescope IDs or telescope_descriptions.
        dl1_images: bool
            load extracted images
        dl1_parameters: bool
            load reconstructed image parameters
        dl1_muons: bool
            load muon ring parameters
        dl2: bool
            load available dl2 stereo parameters
        simulated: bool
            load simulated shower information
        true_images: bool
            load simulated shower images
        true_parameters: bool
            load image parameters obtained from true images
        instrument: bool
            join subarray instrument information to each event
        observation_info: bool
            join observation information to each event
        pointing: bool
            join pointing information to each event

        Returns
        -------
        tables: dict(astropy.io.Table)
            Dictionary of tables organized by telescope ids
            Table with primary index columns "obs_id", "event_id" and "tel_id".
        """
        updated_args = self._check_args(
            dl1_images=dl1_images,
            dl1_parameters=dl1_parameters,
            dl1_muons=dl1_muons,
            dl2=dl2,
            simulated=simulated,
            true_images=true_images,
            true_parameters=true_parameters,
            instrument=instrument,
            observation_info=observation_info,
            pointing=pointing,
        )
        dl1_images = updated_args["dl1_images"]
        dl1_parameters = updated_args["dl1_parameters"]
        dl1_muons = updated_args["dl1_muons"]
        dl2 = updated_args["dl2"]
        simulated = updated_args["simulated"]
        true_images = updated_args["true_images"]
        true_parameters = updated_args["true_parameters"]
        instrument = updated_args["instrument"]
        observation_info = updated_args["observation_info"]
        pointing = updated_args["pointing"]

        if telescopes is None:
            tel_ids = tuple(self.subarray.tel.keys())
        else:
            tel_ids = self.subarray.get_tel_ids(telescopes)

        subarray_events = self.read_subarray_events(
            start=start,
            stop=stop,
            dl2=dl2,
            simulated=simulated,
            observation_info=observation_info,
            keep_order=False,
        )
        self._add_index_if_needed(subarray_events)

        by_id = {}
        for tel_id in tel_ids:
            # no events for this telescope in range start/stop
            table = self._read_telescope_events_for_id(
                tel_id,
                start=start,
                stop=stop,
                dl1_images=dl1_images,
                dl1_parameters=dl1_parameters,
                dl1_muons=dl1_muons,
                dl2=dl2,
                simulated=simulated,
                true_images=true_images,
                true_parameters=true_parameters,
                instrument=instrument,
                pointing=pointing,
            )
            if len(table) > 0:
                by_id[tel_id] = table

        for tel_id in by_id.keys():
            by_id[tel_id] = self._join_subarray_info(
                by_id[tel_id],
                dl2=dl2,
                simulated=simulated,
                observation_info=observation_info,
                subarray_events=subarray_events,
            )
            self._sort_to_original_order(by_id[tel_id], include_tel_id=False)

        return by_id

    def read_telescope_events_by_id_chunked(self, chunk_size, *args, **kwargs):
        """
        Iterate over chunks of telescope events and return a dict of one table per telescope id.

        Parameters
        ----------
        chunk_size: int
            Number of subarray events to load per chunk.
            The telescope tables might be larger or smaller than chunk_size
            depending on the selected telescopes.

        *args, **kwargs are passed to `read_telescope_events_by_id`
        """
        return ChunkIterator(
            self.read_telescope_events_by_id,
            n_total=len(self),
            chunk_size=chunk_size,
            args=args,
            kwargs=kwargs,
        )
