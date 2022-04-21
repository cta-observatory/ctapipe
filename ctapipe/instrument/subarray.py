"""
Description of Arrays or Subarrays of telescopes
"""
from typing import Dict, List, Union
from contextlib import ExitStack
import warnings

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import QTable, Table
from astropy.utils import lazyproperty
import tables
from copy import copy
from itertools import groupby

import ctapipe

from ..coordinates import GroundFrame, CameraFrame
from .telescope import TelescopeDescription
from .camera import CameraDescription, CameraReadout, CameraGeometry
from .optics import OpticsDescription


__all__ = ["SubarrayDescription"]


def _group_consecutives(sequence):
    """
    Turn consequtive lists into ranges (used in SubarrayDescription.info())

    from https://codereview.stackexchange.com/questions/214820/codewars-range-extraction
    """
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

    Parameters
    ----------
    name: str
        name of this subarray
    tel_positions: Dict[Array]
        dict of x,y,z telescope positions on the ground by tel_id. These are
        converted internally to a coordinate in the `~ctapipe.coordinates.GroundFrame`
    tel_descriptions: Dict[TelescopeDescription]
        dict of TelescopeDescriptions by tel_id

    Attributes
    ----------
    name: str
       name of subarray
    tel_coords: astropy.coordinates.SkyCoord
       coordinates of all telescopes
    tels:
       dict of TelescopeDescription for each telescope in the subarray
    tel_ids: np.ndarray
        array of tel_ids
    tel_indices: dict
        dict mapping tel_id to index in array attributes
    """

    def __init__(self, name, tel_positions=None, tel_descriptions=None):
        self.name = name
        self.positions = tel_positions or dict()
        self.tels: Dict[int, TelescopeDescription] = tel_descriptions or dict()

        if self.positions.keys() != self.tels.keys():
            raise ValueError("Telescope ids in positions and descriptions do not match")

    def __str__(self):
        return self.name

    def __repr__(self):
        return "{}(name='{}', num_tels={})".format(
            self.__class__.__name__, self.name, self.num_tels
        )

    @property
    def tel(self):
        """ for backward compatibility"""
        return self.tels

    @property
    def num_tels(self):
        """number of telescopes in this subarray"""
        return len(self.tels)

    def __len__(self):
        return len(self.tels)

    def info(self, printer=print):
        """
        print descriptive info about subarray
        """
        printer(f"Subarray : {self.name}")
        printer(f"Num Tels : {self.num_tels}")
        printer(f"Footprint: {self.footprint:.2f}")
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
        """ returns telescope positions as astropy.coordinates.SkyCoord"""

        pos_x = np.array([p[0].to("m").value for p in self.positions.values()]) * u.m
        pos_y = np.array([p[1].to("m").value for p in self.positions.values()]) * u.m
        pos_z = np.array([p[2].to("m").value for p in self.positions.values()]) * u.m

        return SkyCoord(x=pos_x, y=pos_y, z=pos_z, frame=GroundFrame())

    @lazyproperty
    def tel_ids(self):
        """ telescope IDs as an array"""
        return np.array(list(self.tel.keys()))

    @lazyproperty
    def tel_indices(self):
        """returns dict mapping tel_id to tel_index, useful for unpacking
        lists based on tel_ids into fixed-length arrays"""
        return {tel_id: ii for ii, tel_id in enumerate(self.tels.keys())}

    @lazyproperty
    def tel_index_array(self):
        """
        returns an expanded array that maps tel_id to tel_index. I.e. for a given
        telescope, this array maps the tel_id to a flat index starting at 0 for
        the first telescope. ``tel_index = tel_id_to_index_array[tel_id]``
        If the tel_ids are not contiguous, gaps will be filled in by -1.
        For a more compact representation use the `tel_indices`
        """
        idx = np.zeros(np.max(self.tel_ids) + 1, dtype=int) - 1  # start with -1
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
        tel_ids = np.array(tel_ids, dtype=int, copy=False).ravel()
        return self.tel_index_array[tel_ids]

    def tel_ids_to_mask(self, tel_ids):
        """Convert a list of telescope ids to a boolean mask
        of length ``num_tels`` where the **index** of the telescope
        is set to ``True`` for each tel_id in tel_ids

        Parameters
        ----------
        tel_ids : int or List[int]
            array of tel IDs

        Returns
        -------
        np.array[dtype=bool]:
            Boolean array of length ``num_tels`` with indices of the
            telescopes in ``tel_ids`` set to True.
        """
        mask = np.zeros(self.num_tels, dtype=bool)
        indices = self.tel_ids_to_indices(tel_ids)
        mask[indices] = True
        return mask

    def tel_mask_to_tel_ids(self, tel_mask):
        """
        Convert a boolean mask of selected telescopes to a list of tel_ids.

        Parameters
        ----------
        tel_mask: array-like
            Boolean array of length ``num_tels`` with indices of the
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

        meta = {
            "ORIGIN": "ctapipe.instrument.SubarrayDescription",
            "SUBARRAY": self.name,
            "SOFT_VER": ctapipe.__version__,
            "TAB_TYPE": kind,
        }

        if kind == "subarray":

            unique_types = self.telescope_types

            ids = list(self.tels.keys())
            descs = [str(t) for t in self.tels.values()]
            tel_names = [t.name for t in self.tels.values()]
            tel_types = [t.type for t in self.tels.values()]
            cam_types = [t.camera.camera_name for t in self.tels.values()]
            optics_index = [unique_types.index(t) for t in self.tels.values()]
            camera_index = [
                self.camera_types.index(t.camera) for t in self.tels.values()
            ]
            tel_coords = self.tel_coords

            tab = Table(
                dict(
                    tel_id=np.array(ids, dtype=np.short),
                    pos_x=tel_coords.x,
                    pos_y=tel_coords.y,
                    pos_z=tel_coords.z,
                    name=tel_names,
                    type=tel_types,
                    camera_type=cam_types,
                    camera_index=camera_index,
                    optics_index=optics_index,
                    tel_description=descs,
                )
            )
            tab.meta["TAB_VER"] = "1.0"

        elif kind == "optics":
            unique_types = self.telescope_types

            mirror_area = u.Quantity(
                [t.optics.mirror_area.to_value(u.m ** 2) for t in unique_types],
                u.m ** 2,
            )
            focal_length = u.Quantity(
                [t.optics.equivalent_focal_length.to_value(u.m) for t in unique_types],
                u.m,
            )
            cols = {
                "description": [str(t) for t in unique_types],
                "name": [t.name for t in unique_types],
                "type": [t.type for t in unique_types],
                "mirror_area": mirror_area,
                "num_mirrors": [t.optics.num_mirrors for t in unique_types],
                "num_mirror_tiles": [t.optics.num_mirror_tiles for t in unique_types],
                "equivalent_focal_length": focal_length,
            }
            tab = Table(cols)
            tab.meta["TAB_VER"] = "2.0"
        else:
            raise ValueError(f"Table type '{kind}' not known")

        tab.meta.update(meta)
        return tab

    def select_subarray(self, tel_ids, name=None):
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

        tel_positions = {tid: self.positions[tid] for tid in tel_ids}
        tel_descriptions = {tid: self.tel[tid] for tid in tel_ids}

        if not name:
            tel_ids = sorted(tel_ids)
            name = self.name + "_" + _range_extraction(tel_ids)

        newsub = SubarrayDescription(
            name, tel_positions=tel_positions, tel_descriptions=tel_descriptions
        )
        return newsub

    def peek(self):
        """
        Draw a quick matplotlib plot of the array
        """
        from matplotlib import pyplot as plt
        from astropy.visualization import quantity_support

        types = set(self.tels.values())
        tab = self.to_table()

        plt.figure(figsize=(8, 8))

        with quantity_support():
            for tel_type in types:
                tels = tab[tab["tel_description"] == str(tel_type)]["tel_id"]
                sub = self.select_subarray(tels, name=tel_type)
                tel_coords = sub.tel_coords
                radius = np.array(
                    [
                        np.sqrt(tel.optics.mirror_area / np.pi).value
                        for tel in sub.tels.values()
                    ]
                )

                plt.scatter(
                    tel_coords.x, tel_coords.y, s=radius * 8, alpha=0.5, label=tel_type
                )

            plt.legend(loc="best")
            plt.title(self.name)
            plt.tight_layout()

    @lazyproperty
    def telescope_types(self) -> List[TelescopeDescription]:
        """ list of telescope types in the array"""
        return list({tel for tel in self.tel.values()})

    @lazyproperty
    def camera_types(self) -> List[CameraDescription]:
        """ list of camera types in the array """
        return list({tel.camera for tel in self.tel.values()})

    @lazyproperty
    def optics_types(self) -> List[OpticsDescription]:
        """ list of optics types in the array """
        return list({tel.optics for tel in self.tel.values()})

    def get_tel_ids_for_type(self, tel_type):
        """
        return list of tel_ids that have the given tel_type

        Parameters
        ----------
        tel_type: str or TelescopeDescription
           telescope type string (e.g. 'MST:NectarCam')

        """
        if isinstance(tel_type, TelescopeDescription):
            tel_str = str(tel_type)
        else:
            tel_str = tel_type

        return [id for id, descr in self.tels.items() if str(descr) == tel_str]

    def get_tel_ids(
        self, telescopes: List[Union[int, str, TelescopeDescription]]
    ) -> List[int]:
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

        valid_tel_types = {str(tel_type) for tel_type in self.telescope_types}

        for telescope in telescopes:
            if isinstance(telescope, (int, np.integer)):
                ids.add(telescope)

            if isinstance(telescope, str) and telescope not in valid_tel_types:
                raise ValueError("Invalid telescope type input.")

            ids.update(self.get_tel_ids_for_type(telescope))

        return sorted(ids)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        if self.name != other.name:
            return False

        if self.tels.keys() != other.tels.keys():
            return False

        if self.positions.keys() != other.positions.keys():
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

            if "/configuration/instrument/subarray" in h5file.root:
                if overwrite is False:
                    raise IOError(
                        "File already contains a SubarrayDescription and overwrite=False"
                    )

                h5file.remove_node(
                    "/configuration/instrument/", "subarray", recursive=True
                )
                h5file.remove_node(
                    "/configuration/instrument/", "telescope", recursive=True
                )

            write_table(
                self.to_table(),
                h5file,
                path="/configuration/instrument/subarray/layout",
                mode="a",
            )
            write_table(
                self.to_table(kind="optics"),
                h5file,
                path="/configuration/instrument/telescope/optics",
                mode="a",
            )
            for i, camera in enumerate(self.camera_types):
                write_table(
                    camera.geometry.to_table(),
                    h5file,
                    path=f"/configuration/instrument/telescope/camera/geometry_{i}",
                    mode="a",
                )
                write_table(
                    camera.readout.to_table(),
                    h5file,
                    path=f"/configuration/instrument/telescope/camera/readout_{i}",
                    mode="a",
                )

            h5file.root.configuration.instrument.subarray._v_attrs.name = self.name

    @classmethod
    def from_hdf(cls, path):
        # here to prevent circular import
        from ..io import read_table

        layout = read_table(
            path, "/configuration/instrument/subarray/layout", table_cls=QTable
        )

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
                camera_name=geometry.camera_name, readout=readout, geometry=geometry
            )

        optics_table = read_table(
            path, "/configuration/instrument/telescope/optics", table_cls=QTable
        )
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
                row["name"],
                num_mirrors=row["num_mirrors"],
                equivalent_focal_length=row["equivalent_focal_length"],
                mirror_area=row["mirror_area"],
                num_mirror_tiles=row["num_mirror_tiles"],
            )
            for row in optics_table
        ]

        # give correct frame for the camera to each telescope
        telescope_descriptions = {}
        for row in layout:
            # copy to support different telescopes with same camera geom
            camera = copy(cameras[row["camera_index"]])
            optics = optic_descriptions[row["optics_index"]]
            focal_length = optics.equivalent_focal_length
            camera.geometry.frame = CameraFrame(focal_length=focal_length)

            telescope_descriptions[row["tel_id"]] = TelescopeDescription(
                name=row["name"], tel_type=row["type"], optics=optics, camera=camera
            )

        positions = np.column_stack([layout[f"pos_{c}"] for c in "xyz"])

        name = "Unknown"
        with ExitStack() as stack:
            if not isinstance(path, tables.File):
                path = stack.enter_context(tables.open_file(path, mode="r"))

            attrs = path.root.configuration.instrument.subarray._v_attrs
            if "name" in attrs:
                name = str(attrs.name)

        return cls(
            name=name,
            tel_positions={
                tel_id: pos for tel_id, pos in zip(layout["tel_id"], positions)
            },
            tel_descriptions=telescope_descriptions,
        )
