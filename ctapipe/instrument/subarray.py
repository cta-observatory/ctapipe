"""
Description of Arrays or Subarrays of telescopes
"""

__all__ = ['SubarrayDescription']

from collections import defaultdict

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

import ctapipe
from ..coordinates import GroundFrame


class SubarrayDescription:
    """
    Collects the `TelescopeDescription` of all telescopes along with their
    positions on the ground.

    Parameters
    ----------
    name: str
        name of this subarray
    tel_positions: Dict[Array]
        dict of x,y,z telescope positions on the ground by tel_id. These are
        converted internally to a `SkyCoord` in the `GroundFrame`
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
        self.tels = tel_descriptions or dict()

        if self.positions.keys() != self.tels.keys():
            raise ValueError('Telescope ids in positions and descriptions do not match')

    def __str__(self):
        return self.name

    def __repr__(self):
        return "{}(name='{}', num_tels={})".format(
            self.__class__.__name__,
            self.name,
            self.num_tels,
        )

    @property
    def tel(self):
        """ for backward compatibility"""
        return self.tels

    @property
    def num_tels(self):
        return len(self.tels)

    def __len__(self):
        return len(self.tels)

    def info(self, printer=print):
        """
        print descriptive info about subarray
        """

        teltypes = defaultdict(list)

        for tel_id, desc in self.tels.items():
            teltypes[str(desc)].append(tel_id)

        printer(f"Subarray : {self.name}")
        printer(f"Num Tels : {self.num_tels}")
        printer(f"Footprint: {self.footprint:.2f}")
        printer("")
        printer("                TYPE  Num IDmin  IDmax")
        printer("=====================================")

        for teltype, tels in teltypes.items():
            printer("{:>20s} {:4d} {:4d} ..{:4d}".format(
                teltype, len(tels), min(tels), max(tels)
            ))

    @property
    def tel_coords(self):
        """ returns telescope positions as astropy.coordinates.SkyCoord"""

        pos_x = np.array([p[0].to('m').value
                          for p in self.positions.values()]) * u.m
        pos_y = np.array([p[1].to('m').value
                          for p in self.positions.values()]) * u.m
        pos_z = np.array([p[2].to('m').value
                          for p in self.positions.values()]) * u.m

        return SkyCoord(
            x=pos_x,
            y=pos_y,
            z=pos_z,
            frame=GroundFrame()
        )

    @property
    def tel_ids(self):
        """ telescope IDs as an array"""
        return np.array(list(self.tel.keys()))

    @property
    def tel_indices(self):
        """ returns dict mapping tel_id to tel_index, useful for unpacking
        lists based on tel_ids into fixed-length arrays"""
        return {tel_id: ii for ii, tel_id in enumerate(self.tels.keys())}

    @property
    def footprint(self):
        """area of smallest circle containing array on ground"""
        x = self.tel_coords.x
        y = self.tel_coords.y
        return (np.hypot(x, y).max() ** 2 * np.pi).to('km^2')

    def to_table(self, kind="subarray"):
        """
        export SubarrayDescription information as an `astropy.table.Table`

        Parameters
        ----------
        kind: str
            which table to generate (subarray or optics)
        """

        meta = {
            'ORIGIN': 'ctapipe.inst.SubarrayDescription',
            'SUBARRAY': self.name,
            'SOFT_VER': ctapipe.__version__,
            'TAB_TYPE': kind,
        }

        if kind == 'subarray':

            ids = list(self.tels.keys())
            descs = [str(t) for t in self.tels.values()]
            num_mirrors = [t.optics.num_mirrors for t in self.tels.values()]
            tel_names = [t.name for t in self.tels.values()]
            tel_types = [t.type for t in self.tels.values()]
            cam_types = [t.camera.cam_id for t in self.tels.values()]
            tel_coords = self.tel_coords

            tab = Table(dict(
                tel_id=np.array(ids, dtype=np.short),
                pos_x=tel_coords.x,
                pos_y=tel_coords.y,
                pos_z=tel_coords.z,
                name=tel_names,
                type=tel_types,
                num_mirrors=num_mirrors,
                camera_type=cam_types,
                tel_description=descs,
            ))

        elif kind == 'optics':
            unique_types = set(self.tels.values())

            mirror_area = u.Quantity(
                [t.optics.mirror_area.to_value(u.m**2) for t in unique_types],
                u.m**2,
            )
            focal_length = u.Quantity(
                [t.optics.equivalent_focal_length.to_value(u.m) for t in unique_types],
                u.m,
            )
            cols = {
                'description': [str(t) for t in unique_types],
                'name': [t.name for t in unique_types],
                'type': [t.type for t in unique_types],
                'mirror_area': mirror_area,
                'num_mirrors': [t.optics.num_mirrors for t in unique_types],
                'num_mirror_tiles': [t.optics.num_mirror_tiles for t in unique_types],
                'equivalent_focal_length': focal_length,
            }
            tab = Table(cols)

        else:
            raise ValueError(f"Table type '{kind}' not known")

        tab.meta.update(meta)
        return tab

    def select_subarray(self, name, tel_ids):
        """
        return a new SubarrayDescription that is a sub-array of this one

        Parameters
        ----------
        name: str
            name of new sub-selection
        tel_ids: list(int)
            list of telescope IDs to include in the new subarray

        Returns
        -------
        SubarrayDescription
        """

        tel_positions = {tid: self.positions[tid] for tid in tel_ids}
        tel_descriptions = {tid: self.tel[tid] for tid in tel_ids}

        newsub = SubarrayDescription(name, tel_positions=tel_positions,
                                     tel_descriptions=tel_descriptions)
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
                tels = tab[tab['tel_description'] == str(tel_type)]['tel_id']
                sub = self.select_subarray(tel_type, tels)
                tel_coords = sub.tel_coords
                radius = np.array([
                    np.sqrt(tel.optics.mirror_area / np.pi).value
                    for tel in sub.tels.values()
                ])

                plt.scatter(
                    tel_coords.x,
                    tel_coords.y,
                    s=radius * 8,
                    alpha=0.5,
                    label=tel_type,
                )

            plt.legend(loc='best')
            plt.title(self.name)
            plt.tight_layout()

    @property
    def telescope_types(self):
        """ list of telescope types in the array"""
        return [t.type + ':' + t.camera.cam_id for t in set(self.tel.values())]

    @property
    def camera_types(self):
        """ list of camera types in the array """
        return [t.camera.cam_id for t in set(self.tel.values())]

    @property
    def optics_types(self):
        """ list of optics types in the array """
        return [t.optics for t in set(self.tel.values())]

    def get_tel_ids_for_type(self, tel_type):
        """
        return list of tel_ids that have the given tel_type

        Parameters
        ----------
        tel_type: str
           telescope type string (e.g. 'MST:NectarCam')

        """
        return [id for id, descr in self.tels.items() if str(descr) == tel_type]
