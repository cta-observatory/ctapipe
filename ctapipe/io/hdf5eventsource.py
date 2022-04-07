import astropy.units as u
from astropy.utils.decorators import lazyproperty
import logging
import numpy as np
import tables
from ast import literal_eval

from ..core import Container, Field
from ..instrument import SubarrayDescription
from ..containers import (
    ConcentrationContainer,
    ArrayEventContainer,
    DL1CameraContainer,
    EventIndexContainer,
    CameraHillasParametersContainer,
    HillasParametersContainer,
    IntensityStatisticsContainer,
    LeakageContainer,
    MorphologyContainer,
    SimulationConfigContainer,
    SimulatedShowerContainer,
    SimulatedEventContainer,
    PeakTimeStatisticsContainer,
    CameraTimingParametersContainer,
    TimingParametersContainer,
    TriggerContainer,
    ImageParametersContainer,
    R1CameraContainer,
)
from .eventsource import EventSource
from .hdf5tableio import HDF5TableReader
from .datalevels import DataLevel
from ..utils import IndexFinder


__all__ = ["HDF5EventSource"]


logger = logging.getLogger(__name__)


COMPATIBLE_DL1_VERSIONS = [
    "v1.0.0",
    "v1.0.1",
    "v1.0.2",
    "v1.0.3",
    "v1.1.0",
    "v1.2.0",
    "v2.0.0",
    "v2.1.0",
    "v2.2.0",
    "v3.0.0"
]


def get_hdf5_datalevels(h5file):
    """Get the data levels present in the hdf5 file"""
    datalevels = []

    if "/r1/event/telescope" in h5file.root:
        datalevels.append(DataLevel.R1)

    if "/dl1/event/telescope/images" in h5file.root:
        datalevels.append(DataLevel.DL1_IMAGES)

    if "/dl1/event/telescope/parameters" in h5file.root:
        datalevels.append(DataLevel.DL1_PARAMETERS)

    return tuple(datalevels)


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
        Observation ids of te recorded runs. For unmerged files, this
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
        self._full_subarray_info = SubarrayDescription.from_hdf(self.input_url)

        if self.allowed_tels:
            self._subarray_info = self._full_subarray_info.select_subarray(
                self.allowed_tels
            )
        else:
            self._subarray_info = self._full_subarray_info
        self._simulation_configs = self._parse_simulation_configs()
        self.datamodel_version = self.file_.root._v_attrs[
            "CTA PRODUCT DATA MODEL VERSION"
        ]

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.file_.close()

    @staticmethod
    def is_compatible(file_path):

        with open(file_path, "rb") as f:
            magic_number = f.read(8)

        if magic_number != b"\x89HDF\r\n\x1a\n":
            return False

        with tables.open_file(file_path) as f:
            metadata = f.root._v_attrs
            if "CTA PRODUCT DATA LEVEL" not in metadata._v_attrnames:
                return False

            # we can now read both R1 and DL1
            datalevels = set(literal_eval(metadata["CTA PRODUCT DATA LEVEL"]))
            if not datalevels.intersection(("R1", "DL1_IMAGES", "DL1_PARAMETERS")):
                return False

            if "CTA PRODUCT DATA MODEL VERSION" not in metadata._v_attrnames:
                return False

            version = metadata["CTA PRODUCT DATA MODEL VERSION"]
            if version not in COMPATIBLE_DL1_VERSIONS:
                logger.error(
                    f"File is DL1 file but has unsupported version {version}"
                    f", supported versions are {COMPATIBLE_DL1_VERSIONS}"
                )
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

    @property
    def subarray(self):
        return self._subarray_info

    @lazyproperty
    def datalevels(self):
        return get_hdf5_datalevels(self.file_)

    @lazyproperty
    def obs_ids(self):
        return list(np.unique(self.file_.root.dl1.event.subarray.trigger.col("obs_id")))

    @property
    def simulation_config(self):
        """
        Returns the simulation config(s) as
        a dict mapping obs_id to the respective config.
        """
        return self._simulation_configs

    def _generator(self):
        yield from self._generate_events()

    def __len__(self):
        return len(self.file_.root.dl1.event.subarray.trigger)

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
            container_prefix = ""
            obs_id = Field(-1)

        simulation_configs = {}
        if "simulation" in self.file_.root.configuration:
            reader = HDF5TableReader(self.file_).read(
                "/configuration/simulation/run",
                containers=[SimulationConfigContainer(), ObsIdContainer()],
            )
            for (config, index) in reader:
                simulation_configs[index.obs_id] = config

        return simulation_configs

    def _generate_events(self):
        """
        Yield ArrayEventContainer to iterate through events.
        """
        data = ArrayEventContainer()
        # Maybe take some other metadata, but there are still some 'unknown'
        # written out by the stage1 tool
        data.meta["origin"] = self.file_.root._v_attrs["CTA PROCESS TYPE"]
        data.meta["input_url"] = self.input_url
        data.meta["max_events"] = self.max_events

        self.reader = HDF5TableReader(self.file_)

        if DataLevel.R1 in self.datalevels:
            waveform_readers = {
                tel.name: self.reader.read(
                    f"/r1/event/telescope/{tel.name}", R1CameraContainer()
                )
                for tel in self.file_.root.r1.event.telescope
            }

        if DataLevel.DL1_IMAGES in self.datalevels:
            image_readers = {
                tel.name: self.reader.read(
                    f"/dl1/event/telescope/images/{tel.name}",
                    DL1CameraContainer(),
                    ignore_columns={"parameters"},
                )
                for tel in self.file_.root.dl1.event.telescope.images
            }
            if self.has_simulated_dl1:
                simulated_image_iterators = {
                    tel.name: self.file_.root.simulation.event.telescope.images[
                        tel.name
                    ].iterrows()
                    for tel in self.file_.root.simulation.event.telescope.images
                }

        if DataLevel.DL1_PARAMETERS in self.datalevels:
            param_readers = {
                tel.name: self.reader.read(
                    f"/dl1/event/telescope/parameters/{tel.name}",
                    containers=[
                        (
                            HillasParametersContainer()
                            if (self.datamodel_version >= "v2.1.0")
                            else CameraHillasParametersContainer(prefix="hillas")
                        ),
                        (
                            TimingParametersContainer()
                            if (self.datamodel_version >= "v2.1.0")
                            else CameraTimingParametersContainer(prefix="timing")
                        ),
                        LeakageContainer(),
                        ConcentrationContainer(),
                        MorphologyContainer(),
                        IntensityStatisticsContainer(),
                        PeakTimeStatisticsContainer(),
                    ],
                    prefixes=True,
                )
                for tel in self.file_.root.dl1.event.telescope.parameters
            }
            if self.has_simulated_dl1:
                simulated_param_readers = {
                    tel.name: self.reader.read(
                        f"/simulation/event/telescope/parameters/{tel.name}",
                        containers=[
                            (
                                HillasParametersContainer()
                                if (self.datamodel_version >= "v2.1.0")
                                else CameraHillasParametersContainer(prefix="hillas")
                            ),
                            LeakageContainer(),
                            ConcentrationContainer(),
                            MorphologyContainer(),
                            IntensityStatisticsContainer(),
                        ],
                        prefixes=True,
                    )
                    for tel in self.file_.root.dl1.event.telescope.parameters
                }

        if self.is_simulation:
            # simulated shower wide information
            mc_shower_reader = HDF5TableReader(self.file_).read(
                "/simulation/event/subarray/shower",
                SimulatedShowerContainer(),
                prefixes="true",
            )
            data.simulation = SimulatedEventContainer()

        # Setup iterators for the array events
        events = HDF5TableReader(self.file_).read(
            "/dl1/event/subarray/trigger",
            [TriggerContainer(), EventIndexContainer()],
            ignore_columns={"tel"},
        )

        array_pointing_finder = IndexFinder(
            self.file_.root.dl1.monitoring.subarray.pointing.col("time")
        )

        tel_pointing_finder = {
            tel.name: IndexFinder(tel.col("time"))
            for tel in self.file_.root.dl1.monitoring.telescope.pointing
        }

        for counter, (trigger, index) in enumerate(events):
            data.dl1.tel.clear()
            if self.is_simulation:
                data.simulation.tel.clear()
            data.pointing.tel.clear()
            data.trigger.tel.clear()

            data.count = counter
            data.trigger = trigger
            data.index = index
            data.trigger.tels_with_trigger = (
                self._full_subarray_info.tel_mask_to_tel_ids(
                    data.trigger.tels_with_trigger
                )
            )
            if self.allowed_tels:
                data.trigger.tels_with_trigger = np.intersect1d(
                    data.trigger.tels_with_trigger, np.array(list(self.allowed_tels))
                )

            # Maybe there is a simpler way  to do this
            # Beware: tels_with_trigger contains all triggered telescopes whereas
            # the telescope trigger table contains only the subset of
            # allowed_tels given during the creation of the dl1 file
            for i in self.file_.root.dl1.event.telescope.trigger.where(
                f"(obs_id=={data.index.obs_id}) & (event_id=={data.index.event_id})"
            ):
                if self.allowed_tels and i["tel_id"] not in self.allowed_tels:
                    continue
                if self.datamodel_version == "v1.0.0":
                    data.trigger.tel[i["tel_id"]].time = i["telescopetrigger_time"]
                else:
                    data.trigger.tel[i["tel_id"]].time = i["time"]

            self._fill_array_pointing(data, array_pointing_finder)
            self._fill_telescope_pointing(data, tel_pointing_finder)

            if self.is_simulation:
                data.simulation.shower = next(mc_shower_reader)

            for tel in data.trigger.tel.keys():
                key = f"tel_{tel:03d}"
                if self.allowed_tels and tel not in self.allowed_tels:
                    continue

                if DataLevel.R1 in self.datalevels:
                    data.r1.tel[tel] = next(waveform_readers[key])

                if self.has_simulated_dl1:
                    simulated = data.simulation.tel[tel]

                if DataLevel.DL1_IMAGES in self.datalevels:
                    if key not in image_readers:
                        logger.debug(
                            f"Triggered telescope {tel} is missing "
                            "from the image table."
                        )
                        continue

                    data.dl1.tel[tel] = next(image_readers[key])

                    if self.has_simulated_dl1:
                        if key not in simulated_image_iterators:
                            logger.warning(
                                f"Triggered telescope {tel} is missing "
                                "from the simulated image table, but was present at the "
                                "reconstructed image table."
                            )
                            continue
                        simulated_image_row = next(
                            simulated_image_iterators[f"tel_{tel:03d}"]
                        )
                        simulated.true_image = simulated_image_row["true_image"]

                if DataLevel.DL1_PARAMETERS in self.datalevels:
                    if f"tel_{tel:03d}" not in param_readers.keys():
                        logger.debug(
                            f"Triggered telescope {tel} is missing "
                            "from the parameters table."
                        )
                        continue
                    # Is there a smarter way to unpack this?
                    # Best would probbaly be if we could directly read
                    # into the ImageParametersContainer
                    params = next(param_readers[f"tel_{tel:03d}"])
                    data.dl1.tel[tel].parameters = ImageParametersContainer(
                        hillas=params[0],
                        timing=params[1],
                        leakage=params[2],
                        concentration=params[3],
                        morphology=params[4],
                        intensity_statistics=params[5],
                        peak_time_statistics=params[6],
                    )
                    if self.has_simulated_dl1:
                        if f"tel_{tel:03d}" not in param_readers.keys():
                            logger.debug(
                                f"Triggered telescope {tel} is missing "
                                "from the simulated parameters table, but was "
                                "present at the reconstructed parameters table."
                            )
                            continue
                        simulated_params = next(
                            simulated_param_readers[f"tel_{tel:03d}"]
                        )
                        simulated.true_parameters = ImageParametersContainer(
                            hillas=simulated_params[0],
                            leakage=simulated_params[1],
                            concentration=simulated_params[2],
                            morphology=simulated_params[3],
                            intensity_statistics=simulated_params[4],
                        )

            yield data

    def _fill_array_pointing(self, data, array_pointing_finder):
        """
        Fill the array pointing information of a given event
        """
        # Only unique pointings are stored, so reader.read() wont work as easily
        # Thats why we match the pointings based on trigger time
        closest_time_index = array_pointing_finder.closest(data.trigger.time.mjd)
        array_pointing = self.file_.root.dl1.monitoring.subarray.pointing
        data.pointing.array_azimuth = u.Quantity(
            array_pointing[closest_time_index]["array_azimuth"],
            array_pointing.attrs["array_azimuth_UNIT"],
        )
        data.pointing.array_altitude = u.Quantity(
            array_pointing[closest_time_index]["array_altitude"],
            array_pointing.attrs["array_altitude_UNIT"],
        )
        data.pointing.array_ra = u.Quantity(
            array_pointing[closest_time_index]["array_ra"],
            array_pointing.attrs["array_ra_UNIT"],
        )
        data.pointing.array_dec = u.Quantity(
            array_pointing[closest_time_index]["array_dec"],
            array_pointing.attrs["array_dec_UNIT"],
        )

    def _fill_telescope_pointing(self, data, tel_pointing_finder):
        """
        Fill the telescope pointing information of a given event
        """
        # Same comments as to _fill_array_pointing apply
        for tel in data.trigger.tel.keys():
            if self.allowed_tels and tel not in self.allowed_tels:
                continue
            tel_pointing_table = self.file_.root.dl1.monitoring.telescope.pointing[
                f"tel_{tel:03d}"
            ]
            closest_time_index = tel_pointing_finder[f"tel_{tel:03d}"].closest(
                data.trigger.tel[tel].time
            )
            pointing_telescope = tel_pointing_table
            data.pointing.tel[tel].azimuth = u.Quantity(
                pointing_telescope[closest_time_index]["azimuth"],
                pointing_telescope.attrs["azimuth_UNIT"],
            )
            data.pointing.tel[tel].altitude = u.Quantity(
                pointing_telescope[closest_time_index]["altitude"],
                pointing_telescope.attrs["altitude_UNIT"],
            )
