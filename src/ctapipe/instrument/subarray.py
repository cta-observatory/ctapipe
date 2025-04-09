"""
Description of Arrays or Subarrays of telescopes
"""
import warnings
from collections.abc import Iterable
from contextlib import ExitStack
from copy import copy
from itertools import groupby

import numpy as np
import tables
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates.earth import EarthLocation
from astropy.table import QTable, Table, join
from astropy.utils import lazyproperty

from .. import __version__ as CTAPIPE_VERSION
from ..compat import COPY_IF_NEEDED
from ..coordinates import CameraFrame, GroundFrame
from .camera import CameraDescription, CameraGeometry, CameraReadout
from .optics import FocalLengthKind, OpticsDescription
from .telescope import TelescopeDescription

__all__ = ["SubarrayDescription"]


class UnknownTelescopeID(KeyError):
    """Raised when an unknown telescope id is encountered"""


def _group_consecutives(sequence):
    """
    Turn consecutive lists into ranges (used in SubarrayDescription.info())

    from https://codereview.stackexchange.com/questions/214820/codewars-range-extraction
    """
    sequence = sorted(sequence)
    for _, g in groupby(enumerate(sequence), lambda i_x: i_x[0] - i_x[1]):
        r = [x for _, x in g]
        if len(r) > 2:
            yield f"{r[0]}-{r[-1]}"
        else:
            yield from map(str, r)


def _range_extraction(sequence):
    return ",".join(_group_consecutives(sequence))


class SubarrayDescription:
    """
    Collects the `~ctapipe.instrument.TelescopeDescription` of all telescopes
    along with their positions on the ground.

    Attributes
    ----------
    name: str
       name of subarray
    tel_coords: ctapipe.coordinates.GroundFrame
       coordinates of all telescopes
    tels:
       dict of TelescopeDescription for each telescope in the subarray
    """

    #: Current version number of the format written by `SubarrayDescription.to_hdf`
    CURRENT_TAB_VERSION = "2.0"
    #: Version numbers supported by `SubarrayDescription.from_hdf`
    COMPATIBLE_VERSIONS = {"2.0"}

    def __init__(
        self,
        name,
        tel_positions,
        tel_descriptions,
        reference_location,
    ):
        """
        Initialize a new SubarrayDescription

        Parameters
        ----------
        name : str
            name of this subarray
        tel_positions : dict[int, np.ndarray]
            dict of x,y,z telescope positions on the ground by tel_id. These are
            converted internally to a coordinate in the `~ctapipe.coordinates.GroundFrame`
        tel_descriptions : dict[TelescopeDescription]
            dict of TelescopeDescriptions by tel_id
        reference_location : `astropy.coordinates.EarthLocation`
            EarthLocation of the array reference position, (0, 0, 0) of the
            coordinate system used for `tel_positions`.
        """
        self.name = name
        self.positions = tel_positions or dict()
        self.tels: dict[int, TelescopeDescription] = tel_descriptions or dict()
        self.reference_location = reference_location

        if self.positions.keys() != self.tels.keys():
            raise ValueError("Telescope ids in positions and descriptions do not match")

    def __str__(self):
        return self.name

    def __repr__(self):
        return "{}(name='{}', n_tels={})".format(
            self.__class__.__name__, self.name, self.n_tels
        )

    @property
    def tel(self) -> dict[int, TelescopeDescription]:
        """Dictionary mapping tel_ids to TelescopeDescriptions"""
        return self.tels

    @property
    def n_tels(self):
        """number of telescopes in this subarray"""
        return len(self.tels)

    def __len__(self):
        return len(self.tels)

    def info(self, printer=print):
        """
        print descriptive info about subarray
        """
        printer(f"Subarray : {self.name}")
        printer(f"Num Tels : {self.n_tels}")
        printer(f"Footprint: {self.footprint:.2f}")
        printer(f"Height   : {self.reference_location.geodetic.height:.2f}")
        printer(
            f"Lon/Lat  : {self.reference_location.geodetic.lon}, "
            f"{self.reference_location.geodetic.lat} "
        )
        printer("")

        # print the per-telescope-type informatino:
        n_tels = {}
        tel_ids = {}

        for tel_type in self.telescope_types:
            ids = self.get_tel_ids_for_type(tel_type)
            tel_ids[str(tel_type)] = _range_extraction(ids)
            n_tels[str(tel_type)] = len(ids)

        out_table = Table(
            {
                "Type": list(n_tels.keys()),
                "Count": list(n_tels.values()),
                "Tel IDs": list(tel_ids.values()),
            }
        )
        out_table["Tel IDs"].format = "<s"
        for line in str(out_table).split("\n"):
            printer(line)

    @lazyproperty
    def tel_coords(self):
        """Telescope positions in `~ctapipe.coordinates.GroundFrame`"""

        pos_x = [p[0].to_value(u.m) for p in self.positions.values()]
        pos_y = [p[1].to_value(u.m) for p in self.positions.values()]
        pos_z = [p[2].to_value(u.m) for p in self.positions.values()]

        frame = GroundFrame(reference_location=self.reference_location)

        return SkyCoord(x=pos_x, y=pos_y, z=pos_z, unit=u.m, frame=frame)

    @lazyproperty
    def tel_ids(self):
        """Array of telescope ids in order of telescope indices"""
        return np.array(list(self.tel.keys()))

    @lazyproperty
    def tel_indices(self):
        """dictionary mapping telescope ids to telescope index"""
        return {tel_id: ii for ii, tel_id in enumerate(self.tels.keys())}

    @lazyproperty
    def tel_index_array(self):
        """
        Array of telescope indices that can be indexed by telescope id

        I.e. for a given telescope, this array maps the tel_id to a flat index
        starting at 0 for the first telescope.
        ``tel_index = subarray.tel_index_array[tel_id]``

        If the tel_ids are not contiguous, gaps will be filled in by int minval.
        For a more compact representation use the `tel_indices`
        """
        invalid = np.iinfo(int).min
        idx = np.full(np.max(self.tel_ids) + 1, invalid, dtype=int)
        for key, val in self.tel_indices.items():
            idx[key] = val
        return idx

    def tel_ids_to_indices(self, tel_ids):
        """maps a telescope id (or array of them) to flat indices

        Parameters
        ----------
        tel_ids : int or List[int]
            array of tel IDs

        Returns
        -------
        np.array:
            array of corresponding tel indices
        """
        tel_ids = np.array(tel_ids, dtype=int, copy=COPY_IF_NEEDED).ravel()
        return self.tel_index_array[tel_ids]

    def tel_ids_to_mask(self, tel_ids):
        """Convert a list of telescope ids to a boolean mask
        of length ``n_tels`` where the **index** of the telescope
        is set to ``True`` for each tel_id in tel_ids

        Parameters
        ----------
        tel_ids : int or List[int]
            array of tel IDs

        Returns
        -------
        np.array[dtype=bool]:
            Boolean array of length ``n_tels`` with indices of the
            telescopes in ``tel_ids`` set to True.
        """
        mask = np.zeros(self.n_tels, dtype=bool)
        indices = self.tel_ids_to_indices(tel_ids)
        mask[indices] = True
        return mask

    def tel_mask_to_tel_ids(self, tel_mask):
        """
        Convert a boolean mask of selected telescopes to a list of tel_ids.

        Parameters
        ----------
        tel_mask: array-like
            Boolean array of length ``n_tels`` with indices of the
            telescopes in ``tel_ids`` set to True.
        Returns
        -------
        np.array:
            Array of selected tel_ids
        """
        return self.tel_ids[tel_mask]

    @property
    def footprint(self):
        """area of smallest circle containing array on ground"""
        pos_x = self.tel_coords.x
        pos_y = self.tel_coords.y
        return (np.hypot(pos_x, pos_y).max() ** 2 * np.pi).to("km^2")

    def to_table(self, kind="subarray"):
        """
        export SubarrayDescription information as an `astropy.table.Table`

        Parameters
        ----------
        kind: str
            which table to generate (subarray or optics)
        """

        if kind == "joined":
            table = self.to_table()
            optics = self.to_table(kind="optics")
            optics["optics_index"] = np.arange(len(optics))
            table = join(
                table,
                optics,
                keys="optics_index",
                # conflicts for TAB_VER, TAB_TYPE, not needed here, ignore
                metadata_conflicts="silent",
            )

            table.remove_columns(["optics_index", "camera_index"])
            table.add_index("tel_id")
            return table

        meta = {
            "ORIGIN": "ctapipe.instrument.SubarrayDescription",
            "SUBARRAY": self.name,
            "SOFT_VER": CTAPIPE_VERSION,
            "TAB_TYPE": kind,
        }

        if kind == "subarray":
            if self.reference_location is not None:
                itrs = self.reference_location.itrs
                meta["OBSGEO-X"] = itrs.x.to_value(u.m)
                meta["OBSGEO-Y"] = itrs.y.to_value(u.m)
                meta["OBSGEO-Z"] = itrs.z.to_value(u.m)

            unique_optics = self.optics_types

            ids = list(self.tels.keys())
            descs = [str(t) for t in self.tels.values()]
            tel_names = [t.name for t in self.tels.values()]
            tel_types = [t.optics.size_type.value for t in self.tels.values()]
            cam_names = [t.camera.name for t in self.tels.values()]
            optics_names = [t.optics.name for t in self.tels.values()]
            optics_index = [unique_optics.index(t.optics) for t in self.tels.values()]
            camera_index = [
                self.camera_types.index(t.camera) for t in self.tels.values()
            ]
            tel_coords = self.tel_coords

            tab = Table(
                dict(
                    tel_id=np.array(ids, dtype=np.uint16),
                    name=tel_names,
                    type=tel_types,
                    pos_x=tel_coords.x,
                    pos_y=tel_coords.y,
                    pos_z=tel_coords.z,
                    camera_name=cam_names,
                    optics_name=optics_names,
                    camera_index=camera_index,
                    optics_index=optics_index,
                    tel_description=descs,
                )
            )
            tab.meta["TAB_VER"] = self.CURRENT_TAB_VERSION

        elif kind == "optics":
            unique_optics = self.optics_types

            mirror_area = u.Quantity(
                [o.mirror_area.to_value(u.m**2) for o in unique_optics],
                u.m**2,
            )
            focal_length = u.Quantity(
                [o.equivalent_focal_length.to_value(u.m) for o in unique_optics],
                u.m,
            )
            effective_focal_length = u.Quantity(
                [o.effective_focal_length.to_value(u.m) for o in unique_optics],
                u.m,
            )
            tab = Table(
                {
                    "optics_name": [o.name for o in unique_optics],
                    "size_type": [o.size_type.value for o in unique_optics],
                    "reflector_shape": [o.reflector_shape.value for o in unique_optics],
                    "mirror_area": mirror_area,
                    "n_mirrors": [o.n_mirrors for o in unique_optics],
                    "n_mirror_tiles": [o.n_mirror_tiles for o in unique_optics],
                    "equivalent_focal_length": focal_length,
                    "effective_focal_length": effective_focal_length,
                }
            )
            tab.meta["TAB_VER"] = OpticsDescription.CURRENT_TAB_VERSION
        else:
            raise ValueError(f"Table type '{kind}' not known")

        tab.meta.update(meta)
        return tab

    def select_subarray(self, tel_ids, name=None) -> "SubarrayDescription":
        """
        return a new SubarrayDescription that is a sub-array of this one

        Parameters
        ----------
        tel_ids: list(int)
            list of telescope IDs to include in the new subarray
        name: str
            name of new sub-selection
        Returns
        -------
        SubarrayDescription
        """

        unknown_tel_ids = set(tel_ids).difference(self.tel.keys())
        if len(unknown_tel_ids) > 0:
            known = _range_extraction(self.tel.keys())
            raise UnknownTelescopeID(f"{unknown_tel_ids}, known telescopes: {known}")

        tel_positions = {tid: self.positions[tid] for tid in tel_ids}
        tel_descriptions = {tid: self.tel[tid] for tid in tel_ids}

        if not name:
            name = self.name + "_" + _range_extraction(tel_ids)

        newsub = SubarrayDescription(
            name,
            tel_positions=tel_positions,
            tel_descriptions=tel_descriptions,
            reference_location=self.reference_location,
        )
        return newsub

    def peek(self, ax=None):
        """
        Draw a quick matplotlib plot of the array

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None
            If given, the subarray will be plotted into this ax.
        """
        from matplotlib import pyplot as plt

        from ctapipe.coordinates.ground_frames import EastingNorthingFrame
        from ctapipe.visualization import ArrayDisplay

        if ax is None:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(1, 1, 1)
        else:
            fig = ax.figure

        ad = ArrayDisplay(
            subarray=self, frame=EastingNorthingFrame(), tel_scale=0.75, axes=ax
        )
        ad.add_labels()
        ax.set_title(self.name)

        # This avoids a warning if users have e.g. "figure.constrained_layout"
        # enabled through their matplotlib config. Only call tight layout here
        # if the layout engine has not been set already
        if fig.get_layout_engine() is None:
            fig.tight_layout()

        return ad

    @lazyproperty
    def telescope_types(self) -> tuple[TelescopeDescription]:
        """
        Tuple of unique telescope types in the array

        The entries are ordered by telescope name to be deterministic
        but the order should not be relied on.
        """
        unique_types = set(self.tel.values())
        return tuple(sorted(unique_types, key=lambda tel: tel.name))

    @lazyproperty
    def camera_types(self) -> tuple[CameraDescription]:
        """
        Tuple of unique camera types in the array

        The entries are ordered by camera name to be deterministic
        but the order should not be relied on.
        """
        unique_cameras = {tel.camera for tel in self.tel.values()}
        return tuple(sorted(unique_cameras, key=lambda camera: camera.name))

    @lazyproperty
    def optics_types(self) -> tuple[OpticsDescription]:
        """
        Tuple of unique optics types in the array

        The entries are ordered by optics name to be deterministic
        but the order should not be relied on.
        """
        unique_optics = {tel.optics for tel in self.tel.values()}
        return tuple(sorted(unique_optics, key=lambda optics: optics.name))

    def get_tel_ids_for_type(self, tel_type) -> tuple[int]:
        """
        return tuple of tel_ids that have the given tel_type

        Parameters
        ----------
        tel_type: str or TelescopeDescription
           telescope type string (e.g. 'MST_MST_NectarCam')
        """
        if isinstance(tel_type, TelescopeDescription):
            if tel_type not in self.telescope_types:
                raise ValueError(f"{tel_type} not in subarray: {self.telescope_types}")
            return tuple(
                tel_id for tel_id, descr in self.tels.items() if descr == tel_type
            )
        else:
            valid = {str(tel) for tel in self.telescope_types}
            if tel_type not in valid:
                raise ValueError(f"{tel_type} not in subarray: {valid}")
            return tuple(
                tel_id for tel_id, descr in self.tels.items() if str(descr) == tel_type
            )

    def get_tel_indices_for_type(self, tel_type):
        """
        return tuple of telescope indices that have the given tel_type

        Parameters
        ----------
        tel_type: str or TelescopeDescription
           telescope type string (e.g. 'MST_MST_NectarCam')
        """
        return self.tel_ids_to_indices(self.get_tel_ids_for_type(tel_type))

    def multiplicity(self, tel_mask, tel_type=None):
        """
        Compute multiplicity from a telescope mask

        Parameters
        ----------
        tel_mask : np.ndarray
            Boolean array with last dimension of size n_telescopes
        tel_type : None, str or TelescopeDescription
            If not given, compute multiplicity from all telescopes.
            If given, the multiplicity of the given telescope type will
            be computed.

        Returns
        -------
        multiplicity : int or np.ndarray
            Number of true values for given telescope mask and telescope type
        """

        if tel_type is None:
            return np.count_nonzero(tel_mask, axis=-1)

        return np.count_nonzero(
            tel_mask[..., self.get_tel_indices_for_type(tel_type)], axis=-1
        )

    def get_tel_ids(
        self, telescopes: Iterable[int | str | TelescopeDescription]
    ) -> tuple[int]:
        """
        Convert a list of telescope ids and telescope descriptions to
        a list of unique telescope ids.

        Parameters
        ----------
        telescopes: List[Union[int, str, TelescopeDescription]]
            List of Telescope IDs and descriptions.
            Supported inputs for telescope descriptions are instances of
            `~ctapipe.instrument.TelescopeDescription` as well as their
            string representation.

        Returns
        -------
        tel_ids: List[int]
            List of unique telescope ids matching ``telescopes``
        """
        ids = set()

        # support single telescope element
        if isinstance(telescopes, int | str | TelescopeDescription):
            telescopes = (telescopes,)

        for telescope in telescopes:
            if isinstance(telescope, int | np.integer):
                if telescope not in self.tel:
                    raise ValueError(
                        f"Telescope with tel_id={telescope} not in subarray."
                    )
                ids.add(telescope)
            else:
                ids.update(self.get_tel_ids_for_type(telescope))

        return tuple(sorted(ids))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        if self.name != other.name:
            return False

        if self.tels.keys() != other.tels.keys():
            return False

        if self.positions.keys() != other.positions.keys():
            return False

        if self.reference_location != other.reference_location:
            return False

        for tel_id in self.tels.keys():
            if self.tels[tel_id] != other.tels[tel_id]:
                return False

        for tel_id in self.tels.keys():
            if np.any(self.positions[tel_id] != other.positions[tel_id]):
                return False

        return True

    def to_hdf(self, h5file, overwrite=False, mode="a"):
        """write the SubarrayDescription

        Parameters
        ----------
        h5file : str, bytes, path or tables.File
            Path or already opened tables.File with write permission
        overwrite : False
            If the output path already contains a subarray, by default
            an error will be raised. Set ``overwrite=True`` to overwrite an
            existing subarray. This does not affect other content of the file.
            Use ``mode="w"`` to completely overwrite the output path.
        mode : str
            If h5file is not an already opened file, the output file will
            be opened with the given mode. Must be a mode that enables writing.
        """
        # here to prevent circular import
        from ..io import write_table

        with ExitStack() as stack:
            if not isinstance(h5file, tables.File):
                h5file = stack.enter_context(tables.open_file(h5file, mode=mode))

            if "/configuration/instrument/subarray" in h5file.root and not overwrite:
                raise OSError(
                    "File already contains a SubarrayDescription and overwrite=False"
                )

            subarray_table = self.to_table(kind="subarray")
            subarray_table.meta["name"] = self.name

            if self.reference_location is not None:
                # change FITS convention to something better fitting HDF
                for direction in ("X", "Y", "Z"):
                    fits_key = f"OBSGEO-{direction}"
                    hdf_key = "reference_itrs_" + direction.lower()
                    subarray_table.meta[hdf_key] = subarray_table.meta.pop(fits_key)

            write_table(
                subarray_table,
                h5file,
                path="/configuration/instrument/subarray/layout",
                overwrite=overwrite,
            )
            write_table(
                self.to_table(kind="optics"),
                h5file,
                path="/configuration/instrument/telescope/optics",
                overwrite=overwrite,
            )
            for i, camera in enumerate(self.camera_types):
                write_table(
                    camera.geometry.to_table(),
                    h5file,
                    path=f"/configuration/instrument/telescope/camera/geometry_{i}",
                    overwrite=overwrite,
                )
                write_table(
                    camera.readout.to_table(),
                    h5file,
                    path=f"/configuration/instrument/telescope/camera/readout_{i}",
                    overwrite=overwrite,
                )

    @classmethod
    def from_hdf(cls, path, focal_length_choice=FocalLengthKind.EFFECTIVE):
        # here to prevent circular import
        from ..io import read_table

        if isinstance(focal_length_choice, str):
            focal_length_choice = FocalLengthKind[focal_length_choice.upper()]

        layout = read_table(
            path, "/configuration/instrument/subarray/layout", table_cls=QTable
        )

        version = layout.meta.get("TAB_VER")
        if version not in cls.COMPATIBLE_VERSIONS:
            raise OSError(f"Unsupported version of subarray table: {version}")

        cameras = {}

        # backwards compatibility for older tables, index is the name
        if "camera_index" not in layout.colnames:
            layout["camera_index"] = layout["camera_type"]

        for idx in set(layout["camera_index"]):
            geometry = CameraGeometry.from_table(
                read_table(
                    path, f"/configuration/instrument/telescope/camera/geometry_{idx}"
                )
            )
            readout = CameraReadout.from_table(
                read_table(
                    path, f"/configuration/instrument/telescope/camera/readout_{idx}"
                )
            )
            cameras[idx] = CameraDescription(
                name=geometry.name, readout=readout, geometry=geometry
            )

        optics_table = read_table(
            path, "/configuration/instrument/telescope/optics", table_cls=QTable
        )

        optics_version = optics_table.meta.get("TAB_VER")
        if optics_version not in OpticsDescription.COMPATIBLE_VERSIONS:
            raise OSError(f"Unsupported version of optics table: {optics_version}")

        # for backwards compatibility
        # if optics_index not in table, guess via telescope_description string
        # might not result in correct array when there are duplicated telescope_description
        # strings
        if "optics_index" not in layout.colnames:
            descriptions = list(optics_table["description"])
            layout["optics_index"] = [
                descriptions.index(tel) for tel in layout["tel_description"]
            ]
            if len(descriptions) != len(set(descriptions)):
                warnings.warn(
                    "Array contains different telescopes with the same description string"
                    " representation, optics descriptions will be incorrect for some of the telescopes."
                    " Reprocessing the data with ctapipe >= 0.12 will fix this problem."
                )

        optic_descriptions = [
            OpticsDescription(
                name=str(row["optics_name"]),
                size_type=row["size_type"],
                reflector_shape=row["reflector_shape"],
                n_mirrors=row["n_mirrors"],
                equivalent_focal_length=row["equivalent_focal_length"],
                effective_focal_length=row["effective_focal_length"],
                mirror_area=row["mirror_area"],
                n_mirror_tiles=row["n_mirror_tiles"],
            )
            for row in optics_table
        ]

        # give correct frame for the camera to each telescope
        telescope_descriptions = {}
        for row in layout:
            # copy to support different telescopes with same camera geom
            camera = copy(cameras[row["camera_index"]])
            optics = optic_descriptions[row["optics_index"]]

            if focal_length_choice is FocalLengthKind.EFFECTIVE:
                focal_length = optics.effective_focal_length
                if np.isnan(focal_length.value):
                    raise RuntimeError(
                        "`focal_length_choice` was set to 'EFFECTIVE', but the"
                        " effective focal length was not present in the file. "
                        " Set `focal_length_choice='EQUIVALENT'` or make sure"
                        " input files contain the effective focal length"
                    )
            elif focal_length_choice is FocalLengthKind.EQUIVALENT:
                focal_length = optics.equivalent_focal_length
            else:
                raise ValueError(f"Invalid focal length choice: {focal_length_choice}")

            camera.geometry.frame = CameraFrame(focal_length=focal_length)
            telescope_descriptions[row["tel_id"]] = TelescopeDescription(
                name=str(row["name"]), optics=optics, camera=camera
            )

        positions = np.column_stack([layout[f"pos_{c}"] for c in "xyz"])

        name = layout.meta.get("SUBARRAY", "Unknown")

        if "reference_itrs_x" in layout.meta:
            reference_location = EarthLocation(
                x=layout.meta["reference_itrs_x"] * u.m,
                y=layout.meta["reference_itrs_y"] * u.m,
                z=layout.meta["reference_itrs_z"] * u.m,
            )
        else:
            # "null island" dummy location for backwards compatibility with old files
            reference_location = EarthLocation(
                lon=0 * u.deg,
                lat=0 * u.deg,
                height=0 * u.m,
            )

        return cls(
            name=str(name),
            tel_positions={
                tel_id: pos for tel_id, pos in zip(layout["tel_id"], positions)
            },
            tel_descriptions=telescope_descriptions,
            reference_location=reference_location,
        )

    @staticmethod
    def read(path, **kwargs):
        """Read subarray from path

        This uses the `~ctapipe.io.EventSource` mechanism, so it should be
        able to read a subarray from any file supported by ctapipe or an
        installed io plugin.

        kwargs are passed to the event source
        """
        # here to prevent circular import
        from ..io import EventSource

        with EventSource(path, **kwargs) as s:
            return s.subarray
