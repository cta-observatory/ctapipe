import logging
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import tables
from astropy.table import QTable
from astropy.utils.decorators import lazyproperty

from ..atmosphere import AtmosphereDensityProfile
from ..containers import (
    ArrayEventContainer,
    CameraHillasParametersContainer,
    CameraTimingParametersContainer,
    ConcentrationContainer,
    CoordinateFrameType,
    DispContainer,
    DL1CameraContainer,
    EventIndexContainer,
    HillasParametersContainer,
    ImageParametersContainer,
    IntensityStatisticsContainer,
    LeakageContainer,
    MorphologyContainer,
    MuonEfficiencyContainer,
    MuonParametersContainer,
    MuonRingContainer,
    MuonTelescopeContainer,
    ObservationBlockContainer,
    ParticleClassificationContainer,
    PeakTimeStatisticsContainer,
    R1CameraContainer,
    ReconstructedEnergyContainer,
    ReconstructedGeometryContainer,
    SchedulingBlockContainer,
    SimulatedEventContainer,
    SimulatedShowerContainer,
    SimulatedShowerDistribution,
    SimulationConfigContainer,
    TelescopeImpactParameterContainer,
    TelescopePointingContainer,
    TelescopeTriggerContainer,
    TelEventIndexContainer,
    TimingParametersContainer,
    TriggerContainer,
)
from ..core import Container, Field, Provenance
from ..core.traits import UseEnum
from ..instrument import SubarrayDescription
from ..instrument.optics import FocalLengthKind
from ..monitoring.interpolation import PointingInterpolator
from ..utils import IndexFinder
from .astropy_helpers import read_table
from .datalevels import DataLevel
from .eventsource import EventSource
from .hdf5tableio import HDF5TableReader
from .metadata import _read_reference_metadata_hdf5
from .tableloader import DL2_SUBARRAY_GROUP, DL2_TELESCOPE_GROUP, POINTING_GROUP

__all__ = ["HDF5EventSource"]


logger = logging.getLogger(__name__)


DL2_CONTAINERS = {
    "energy": ReconstructedEnergyContainer,
    "geometry": ReconstructedGeometryContainer,
    "classification": ParticleClassificationContainer,
    "impact": TelescopeImpactParameterContainer,
    "disp": DispContainer,
}


COMPATIBLE_DATA_MODEL_VERSIONS = [
    "v4.0.0",
    "v5.0.0",
    "v5.1.0",
    "v6.0.0",
]


def get_hdf5_datalevels(h5file: tables.File | str | Path):
    """Get the data levels present in the hdf5 file"""
    datalevels = []

    with ExitStack() as stack:
        if not isinstance(h5file, tables.File):
            h5file = stack.enter_context(tables.open_file(h5file))

        if "/r1/event/telescope" in h5file.root:
            datalevels.append(DataLevel.R1)

        if "/dl1/event/telescope/images" in h5file.root:
            datalevels.append(DataLevel.DL1_IMAGES)

        if "/dl1/event/telescope/parameters" in h5file.root:
            datalevels.append(DataLevel.DL1_PARAMETERS)

        if "/dl1/event/telescope/muon" in h5file.root:
            datalevels.append(DataLevel.DL1_MUON)

        if "/dl2" in h5file.root:
            datalevels.append(DataLevel.DL2)

    return tuple(datalevels)


def read_atmosphere_density_profile(
    h5file: tables.File, path="/simulation/service/atmosphere_density_profile"
):
    """return a subclass of AtmosphereDensityProfile by
    reading a table in a h5 file

    Parameters
    ----------
    h5file: tables.File
        file handle of HDF5 file
    path: str
        path in the file where the serialized model is stored as an
        astropy.table.Table

    Returns
    -------
    AtmosphereDensityProfile:
        subclass depending on type of table
    """

    if path not in h5file:
        return None

    table = read_table(h5file=h5file, path=path)
    return AtmosphereDensityProfile.from_table(table)


class HDF5EventSource(EventSource):
    """
    Event source for files in the ctapipe DL1 format.
    For general information about the concept of event sources,
    take a look at the parent class ctapipe.io.EventSource.

    To use this event source, create an instance of this class
    specifying the file to be read.

    Looping over the EventSource yields events from the _generate_events
    method. An event equals an ArrayEventContainer instance.
    See ctapipe.containers.ArrayEventContainer for details.

    Attributes
    ----------
    input_url: str
        Path to the input event file.
    file: tables.File
        File object
    obs_ids: list
        Observation ids of the recorded runs. For unmerged files, this
        should only contain a single number.
    subarray: ctapipe.instrument.SubarrayDescription
        The subarray configuration of the recorded run.
    datalevels: Tuple
        One or both of ctapipe.io.datalevels.DataLevel.DL1_IMAGES
        and ctapipe.io.datalevels.DataLevel.DL1_PARAMETERS
        depending on the information present in the file.
    is_simulation: Boolean
        Whether the events are simulated or observed.
    simulation_configs: Dict
        Mapping of obs_id to ctapipe.containers.SimulationConfigContainer
        if the file contains simulated events.
    has_simulated_dl1: Boolean
        Whether the file contains simulated camera images and/or
        image parameters evaluated on these.
    """

    focal_length_choice = UseEnum(
        FocalLengthKind,
        default_value=FocalLengthKind.EFFECTIVE,
        help=(
            "If both nominal and effective focal lengths are available, "
            " which one to use for the `~ctapipe.coordinates.CameraFrame` attached"
            " to the `~ctapipe.instrument.CameraGeometry` instances in the"
            " `~ctapipe.instrument.SubarrayDescription` which will be used in"
            " CameraFrame to TelescopeFrame coordinate transforms."
            " The 'nominal' focal length is the one used during "
            " the simulation, the 'effective' focal length is computed using specialized "
            " ray-tracing from a point light source"
        ),
    ).tag(config=True)

    def __init__(self, input_url=None, config=None, parent=None, **kwargs):
        """
        EventSource for dl1 files in the standard DL1 data format

        Parameters:
        -----------
        input_url : str
            Path of the file to load
        config : traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            Set to None if no configuration to pass.
        parent:
            Parent from which the config is used. Mutually exclusive with config
        kwargs
        """
        super().__init__(input_url=input_url, config=config, parent=parent, **kwargs)

        self.file_ = tables.open_file(self.input_url)
        meta = _read_reference_metadata_hdf5(self.file_)
        Provenance().add_input_file(
            str(self.input_url), role="Event", reference_meta=meta
        )

        self._full_subarray = SubarrayDescription.from_hdf(
            self.input_url,
            focal_length_choice=self.focal_length_choice,
        )

        if self.allowed_tels:
            self._subarray = self._full_subarray.select_subarray(self.allowed_tels)
        else:
            self._subarray = self._full_subarray
        self._simulation_configs = self._parse_simulation_configs()
        (
            self._scheduling_block,
            self._observation_block,
        ) = self._parse_sb_and_ob_configs()

        version = self.file_.root._v_attrs["CTA PRODUCT DATA MODEL VERSION"]
        self.datamodel_version = tuple(map(int, version.lstrip("v").split(".")))
        self._obs_ids = tuple(
            self.file_.root.configuration.observation.observation_block.col("obs_id")
        )
        pointing_key = "/configuration/telescope/pointing"
        # for ctapipe <0.21
        legacy_pointing_key = "/dl1/monitoring/telescope/pointing"
        self._legacy_tel_pointing_finders = {}
        self._legacy_tel_pointing_tables = {}

        self._constant_telescope_pointing = {}
        if pointing_key in self.file_.root:
            for h5table in self.file_.root[pointing_key]._f_iter_nodes("Table"):
                tel_id = int(h5table._v_name.partition("tel_")[-1])
                table = QTable(read_table(self.file_, h5table._v_pathname), copy=False)
                table.add_index("obs_id")
                self._constant_telescope_pointing[tel_id] = table
        elif legacy_pointing_key in self.file_.root:
            self.log.info(
                "Found file written using ctapipe<0.21, using legacy pointing information"
            )
            for node in self.file_.root[legacy_pointing_key]:
                tel_id = int(node.name.removeprefix("tel_"))
                table = QTable(read_table(self.file_, node._v_pathname), copy=False)
                self._legacy_tel_pointing_tables[tel_id] = table
                self._legacy_tel_pointing_finders[tel_id] = IndexFinder(
                    table["time"].mjd
                )

        self._simulated_shower_distributions = (
            self._read_simulated_shower_distributions()
        )

    def _read_simulated_shower_distributions(self):
        key = "/simulation/service/shower_distribution"
        if key not in self.file_.root:
            return {}

        reader = HDF5TableReader(self.file_).read(
            key, containers=SimulatedShowerDistribution
        )
        return {dist.obs_id: dist for dist in reader}

    @property
    def simulated_shower_distributions(self):
        return self._simulated_shower_distributions

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.file_.close()

    @staticmethod
    def is_compatible(file_path):
        path = Path(file_path).expanduser()
        if not path.is_file():
            return False

        with path.open("rb") as f:
            magic_number = f.read(8)

        if magic_number != b"\x89HDF\r\n\x1a\n":
            return False

        with tables.open_file(path) as f:
            metadata = f.root._v_attrs

            if "CTA PRODUCT DATA MODEL VERSION" not in metadata._v_attrnames:
                return False

            version = metadata["CTA PRODUCT DATA MODEL VERSION"]
            if version not in COMPATIBLE_DATA_MODEL_VERSIONS:
                logger.error(
                    "File is a ctapipe HDF5 file but has unsupported data model"
                    f" version {version}"
                    f", supported versions are {COMPATIBLE_DATA_MODEL_VERSIONS}."
                    " You may need to downgrade ctapipe (if the file version is older)"
                    ", update ctapipe (if the file version is newer) or"
                    " reproduce the file with your current ctapipe version."
                )
                return False

            if "CTA PRODUCT DATA LEVELS" not in metadata._v_attrnames:
                return False

            # we can now read both R1 and DL1
            has_muons = "/dl1/event/telescope/muon" in f.root
            datalevels = set(metadata["CTA PRODUCT DATA LEVELS"].split(","))
            datalevels = datalevels & {
                "R1",
                "DL1_IMAGES",
                "DL1_PARAMETERS",
                "DL2",
                "DL1_MUON",
            }
            if not datalevels and not has_muons:
                return False

        return True

    @property
    def is_simulation(self):
        """
        True for files with a simulation group at the root of the file.
        """
        return "simulation" in self.file_.root

    @property
    def has_simulated_dl1(self):
        """
        True for files with telescope-wise event information in the simulation group
        """
        if self.is_simulation:
            if "telescope" in self.file_.root.simulation.event:
                return True
        return False

    @lazyproperty
    def has_muon_parameters(self):
        """
        True for files that contain muon parameters
        """
        return "/dl1/event/telescope/muon" in self.file_.root

    @property
    def subarray(self):
        return self._subarray

    @lazyproperty
    def datalevels(self):
        return get_hdf5_datalevels(self.file_)

    @lazyproperty
    def atmosphere_density_profile(self) -> AtmosphereDensityProfile:
        return read_atmosphere_density_profile(self.file_)

    @property
    def obs_ids(self):
        return self._obs_ids

    @property
    def scheduling_blocks(self) -> dict[int, SchedulingBlockContainer]:
        return self._scheduling_block

    @property
    def observation_blocks(self) -> dict[int, ObservationBlockContainer]:
        return self._observation_block

    @property
    def simulation_config(self) -> dict[int, SimulationConfigContainer]:
        """
        Returns the simulation config(s) as
        a dict mapping obs_id to the respective config.
        """
        return self._simulation_configs

    def __len__(self):
        n_events = len(self.file_.root.dl1.event.subarray.trigger)
        if self.max_events is not None:
            return min(n_events, self.max_events)
        return n_events

    def _parse_simulation_configs(self):
        """
        Construct a dict of SimulationConfigContainers from the
        self.file_.root.configuration.simulation.run.
        These are used to match the correct header to each event
        """

        # Just returning next(reader) would work as long as there are no merged files
        # The reader ignores obs_id making the setup somewhat tricky
        # This is ugly but supports multiple headers so each event can have
        # the correct mcheader assigned by matching the obs_id
        # Alternatively this becomes a flat list
        # and the obs_id matching part needs to be done in _generate_events()
        class ObsIdContainer(Container):
            default_prefix = ""
            obs_id = Field(-1)

        if "simulation" in self.file_.root.configuration:
            reader = HDF5TableReader(self.file_).read(
                "/configuration/simulation/run",
                containers=(SimulationConfigContainer, ObsIdContainer),
            )
            return {index.obs_id: config for (config, index) in reader}
        else:
            return {}

    def _parse_sb_and_ob_configs(self):
        """read Observation and Scheduling block configurations"""

        sb_reader = HDF5TableReader(self.file_).read(
            "/configuration/observation/scheduling_block",
            containers=SchedulingBlockContainer,
        )

        scheduling_blocks = {sb.sb_id: sb for sb in sb_reader}

        ob_reader = HDF5TableReader(self.file_).read(
            "/configuration/observation/observation_block",
            containers=ObservationBlockContainer,
        )
        observation_blocks = {ob.obs_id: ob for ob in ob_reader}

        return scheduling_blocks, observation_blocks

    def _is_hillas_in_camera_frame(self):
        parameters_group = self.file_.root.dl1.event.telescope.parameters
        telescope_tables = parameters_group._v_children.values()

        # in case of no parameters, it doesn't matter, we just return False
        if len(telescope_tables) == 0:
            return False

        # check the first telescope table
        one_telescope = parameters_group._v_children.values()[0]
        return "camera_frame_hillas_intensity" in one_telescope.colnames

    def _generator(self):
        """
        Yield ArrayEventContainer to iterate through events.
        """
        self.reader = HDF5TableReader(self.file_)

        if DataLevel.R1 in self.datalevels:
            waveform_readers = {
                table.name: self.reader.read(
                    f"/r1/event/telescope/{table.name}", R1CameraContainer
                )
                for table in self.file_.root.r1.event.telescope
            }

        if DataLevel.DL1_IMAGES in self.datalevels:
            ignore_columns = {"parameters"}

            # if there are no parameters, there are no image_mask, avoids warnings
            if DataLevel.DL1_PARAMETERS not in self.datalevels:
                ignore_columns.add("image_mask")

            image_readers = {
                table.name: self.reader.read(
                    f"/dl1/event/telescope/images/{table.name}",
                    DL1CameraContainer,
                    ignore_columns=ignore_columns,
                )
                for table in self.file_.root.dl1.event.telescope.images
            }
            if self.has_simulated_dl1:
                simulated_image_iterators = {
                    table.name: self.file_.root.simulation.event.telescope.images[
                        table.name
                    ].iterrows()
                    for table in self.file_.root.simulation.event.telescope.images
                }

        if DataLevel.DL1_PARAMETERS in self.datalevels:
            hillas_cls = HillasParametersContainer
            timing_cls = TimingParametersContainer
            hillas_prefix = "hillas"
            timing_prefix = "timing"

            if self._is_hillas_in_camera_frame():
                hillas_cls = CameraHillasParametersContainer
                timing_cls = CameraTimingParametersContainer
                hillas_prefix = "camera_frame_hillas"
                timing_prefix = "camera_frame_timing"

            param_readers = {
                table.name: self.reader.read(
                    f"/dl1/event/telescope/parameters/{table.name}",
                    containers=(
                        hillas_cls,
                        timing_cls,
                        LeakageContainer,
                        ConcentrationContainer,
                        MorphologyContainer,
                        IntensityStatisticsContainer,
                        PeakTimeStatisticsContainer,
                    ),
                    prefixes=[
                        hillas_prefix,
                        timing_prefix,
                        "leakage",
                        "concentration",
                        "morphology",
                        "intensity",
                        "peak_time",
                    ],
                )
                for table in self.file_.root.dl1.event.telescope.parameters
            }
            if self.has_simulated_dl1:
                simulated_param_readers = {
                    table.name: self.reader.read(
                        f"/simulation/event/telescope/parameters/{table.name}",
                        containers=[
                            hillas_cls,
                            LeakageContainer,
                            ConcentrationContainer,
                            MorphologyContainer,
                            IntensityStatisticsContainer,
                        ],
                        prefixes=[
                            f"true_{hillas_prefix}",
                            "true_leakage",
                            "true_concentration",
                            "true_morphology",
                            "true_intensity",
                        ],
                    )
                    for table in self.file_.root.dl1.event.telescope.parameters
                }

        if self.has_muon_parameters:
            muon_readers = {
                table.name: self.reader.read(
                    f"/dl1/event/telescope/muon/{table.name}",
                    containers=[
                        MuonRingContainer,
                        MuonParametersContainer,
                        MuonEfficiencyContainer,
                    ],
                )
                for table in self.file_.root.dl1.event.telescope.muon
            }

        dl2_readers = {}
        if DL2_SUBARRAY_GROUP in self.file_.root:
            dl2_group = self.file_.root[DL2_SUBARRAY_GROUP]

            for kind, group in dl2_group._v_children.items():
                try:
                    container = DL2_CONTAINERS[kind]
                except KeyError:
                    self.log.warning("Unknown DL2 stereo group %s", kind)
                    continue

                dl2_readers[kind] = {
                    algorithm: HDF5TableReader(self.file_).read(
                        table._v_pathname,
                        containers=container,
                        prefixes=(algorithm,),
                    )
                    for algorithm, table in group._v_children.items()
                }

        dl2_tel_readers = {}
        if DL2_TELESCOPE_GROUP in self.file_.root:
            dl2_group = self.file_.root[DL2_TELESCOPE_GROUP]

            for kind, group in dl2_group._v_children.items():
                try:
                    container = DL2_CONTAINERS[kind]
                except KeyError:
                    self.log.warning("Unknown DL2 telescope group %s", kind)
                    continue

                dl2_tel_readers[kind] = {}
                for algorithm, algorithm_group in group._v_children.items():
                    dl2_tel_readers[kind][algorithm] = {
                        key: HDF5TableReader(self.file_).read(
                            table._v_pathname,
                            containers=container,
                        )
                        for key, table in algorithm_group._v_children.items()
                    }

        true_impact_readers = {}
        if self.is_simulation:
            # simulated shower wide information
            mc_shower_reader = HDF5TableReader(self.file_).read(
                "/simulation/event/subarray/shower",
                SimulatedShowerContainer,
                prefixes="true",
            )
            if "impact" in self.file_.root.simulation.event.telescope:
                true_impact_readers = {
                    table.name: self.reader.read(
                        f"/simulation/event/telescope/impact/{table.name}",
                        containers=TelescopeImpactParameterContainer,
                        prefixes=["true_impact"],
                    )
                    for table in self.file_.root.simulation.event.telescope.impact
                }

        # Setup iterators for the array events
        events = HDF5TableReader(self.file_).read(
            "/dl1/event/subarray/trigger",
            [TriggerContainer, EventIndexContainer],
            ignore_columns={"tel"},
        )
        telescope_trigger_reader = HDF5TableReader(self.file_).read(
            "/dl1/event/telescope/trigger",
            [TelEventIndexContainer, TelescopeTriggerContainer],
            ignore_columns={"trigger_pixels"},
        )

        pointing_interpolator = None
        if POINTING_GROUP in self.file_.root:
            pointing_interpolator = PointingInterpolator(
                h5file=self.file_,
                parent=self,
            )

        counter = 0
        for trigger, index in events:
            data = ArrayEventContainer(
                trigger=trigger,
                count=counter,
                index=index,
                simulation=SimulatedEventContainer() if self.is_simulation else None,
            )
            # Maybe take some other metadata, but there are still some 'unknown'
            # written out by the stage1 tool
            data.meta["origin"] = self.file_.root._v_attrs["CTA PROCESS TYPE"]
            data.meta["input_url"] = self.input_url
            data.meta["max_events"] = self.max_events

            data.trigger.tels_with_trigger = self._full_subarray.tel_mask_to_tel_ids(
                data.trigger.tels_with_trigger
            )
            full_tels_with_trigger = data.trigger.tels_with_trigger.copy()
            if self.allowed_tels:
                data.trigger.tels_with_trigger = np.intersect1d(
                    data.trigger.tels_with_trigger, np.array(list(self.allowed_tels))
                )

            # the telescope trigger table contains triggers for all telescopes
            # that participated in the event, so we need to read a row for each
            # of them, ignoring the ones not in allowed_tels after reading
            for tel_id in full_tels_with_trigger:
                tel_index, tel_trigger = next(telescope_trigger_reader)

                if self.allowed_tels and tel_id not in self.allowed_tels:
                    continue

                data.trigger.tel[tel_index.tel_id] = tel_trigger

            if self.is_simulation:
                data.simulation.shower = next(mc_shower_reader)

            for kind, readers in dl2_readers.items():
                c = getattr(data.dl2.stereo, kind)
                for algorithm, reader in readers.items():
                    c[algorithm] = next(reader)

            # this needs to stay *after* reading the telescope trigger table
            # and after reading all subarray event information, so that we don't
            # go out of sync
            if len(data.trigger.tels_with_trigger) == 0:
                continue

            self._fill_array_pointing(data)
            self._fill_telescope_pointing(data, pointing_interpolator)

            for tel_id in data.trigger.tel.keys():
                key = f"tel_{tel_id:03d}"

                if self.allowed_tels and tel_id not in self.allowed_tels:
                    continue

                if key in true_impact_readers:
                    data.simulation.tel[tel_id].impact = next(true_impact_readers[key])

                if DataLevel.R1 in self.datalevels:
                    data.r1.tel[tel_id] = next(waveform_readers[key])

                    # TODO: Delete if we drop support for datamodel versions < v6.0.0
                    r1_waveform = data.r1.tel[tel_id].waveform
                    if r1_waveform.ndim == 2:
                        data.r1.tel[tel_id].waveform = r1_waveform[np.newaxis, ...]

                if self.has_simulated_dl1:
                    simulated = data.simulation.tel[tel_id]

                if DataLevel.DL1_IMAGES in self.datalevels:
                    data.dl1.tel[tel_id] = next(image_readers[key])

                    if self.has_simulated_dl1:
                        simulated_image_row = next(simulated_image_iterators[key])
                        simulated.true_image = simulated_image_row["true_image"]
                        simulated.true_image_sum = simulated_image_row["true_image_sum"]

                if DataLevel.DL1_PARAMETERS in self.datalevels:
                    # Is there a smarter way to unpack this?
                    # Best would probably be if we could directly read
                    # into the ImageParametersContainer
                    params = next(param_readers[key])
                    data.dl1.tel[tel_id].parameters = ImageParametersContainer(
                        hillas=params[0],
                        timing=params[1],
                        leakage=params[2],
                        concentration=params[3],
                        morphology=params[4],
                        intensity_statistics=params[5],
                        peak_time_statistics=params[6],
                    )
                    if self.has_simulated_dl1:
                        simulated_params = next(simulated_param_readers[key])
                        simulated.true_parameters = ImageParametersContainer(
                            hillas=simulated_params[0],
                            leakage=simulated_params[1],
                            concentration=simulated_params[2],
                            morphology=simulated_params[3],
                            intensity_statistics=simulated_params[4],
                        )

                if self.has_muon_parameters:
                    ring, parameters, efficiency = next(muon_readers[key])
                    data.muon.tel[tel_id] = MuonTelescopeContainer(
                        ring=ring,
                        parameters=parameters,
                        efficiency=efficiency,
                    )

                for kind, algorithms in dl2_tel_readers.items():
                    c = getattr(data.dl2.tel[tel_id], kind)
                    for algorithm, readers in algorithms.items():
                        c[algorithm] = next(readers[key])

                        # change prefix to new data model
                        if kind == "impact" and self.datamodel_version == (4, 0, 0):
                            prefix = f"{algorithm}_tel_{c[algorithm].default_prefix}"
                            c[algorithm].prefix = prefix

            yield data
            counter += 1

    def _fill_array_pointing(self, event):
        """
        Fill the array pointing information of a given event
        """
        obs_id = event.index.obs_id
        ob = self.observation_blocks[obs_id]
        frame = ob.subarray_pointing_frame
        if frame is CoordinateFrameType.ALTAZ:
            event.pointing.array_azimuth = ob.subarray_pointing_lon
            event.pointing.array_altitude = ob.subarray_pointing_lat
        elif frame is CoordinateFrameType.ICRS:
            event.pointing.array_ra = ob.subarray_pointing_lon
            event.pointing.array_dec = ob.subarray_pointing_lat
        else:
            raise ValueError(f"Unsupported pointing frame: {frame}")

    def _fill_telescope_pointing(self, event, tel_pointing_interpolator=None):
        """
        Fill the telescope pointing information of a given event
        """
        if tel_pointing_interpolator is not None:
            for tel_id, trigger in event.trigger.tel.items():
                alt, az = tel_pointing_interpolator(tel_id, trigger.time)
                event.pointing.tel[tel_id] = TelescopePointingContainer(
                    altitude=alt,
                    azimuth=az,
                )
            return

        for tel_id in event.trigger.tels_with_trigger:
            if (
                tel_pointing := self._constant_telescope_pointing.get(tel_id)
            ) is not None:
                current = tel_pointing.loc[event.index.obs_id]
                event.pointing.tel[tel_id] = TelescopePointingContainer(
                    altitude=current["telescope_pointing_altitude"],
                    azimuth=current["telescope_pointing_azimuth"],
                )
            elif (finder := self._legacy_tel_pointing_finders.get(tel_id)) is not None:
                index = finder.closest(event.trigger.time.mjd)
                row = self._legacy_tel_pointing_tables[tel_id][index]
                event.pointing.tel[tel_id] = TelescopePointingContainer(
                    altitude=row["altitude"],
                    azimuth=row["azimuth"],
                )
