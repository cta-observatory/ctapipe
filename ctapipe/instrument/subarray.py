"""
Description of Arrays or Subarrays of telescopes
"""

__all__ = ['SubarrayDescription']

import warnings
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

        self.name = name  #: name of telescope
        self.positions = tel_positions or dict()
        self.tels = tel_descriptions or dict()

        assert set(self.positions.keys()) == set(self.tels.keys())

    def __str__(self):
        return self.name

    def __repr__(self):
        return "{}(name='{}', num_tels={})".format(
            self.__class__.__name__,
            self.name,
            self.num_tels)

    @property
    def tel(self):
        """ for backward compatibility"""
        return self.tels

    @property
    def num_tels(self):
        return len(self.tels)

    def info(self, printer=print):
        """
        print descriptive info about subarray
        """

        teltypes = defaultdict(list)

        for tel_id, desc in self.tels.items():
            teltypes[str(desc)].append(tel_id)

        printer("Subarray : {}".format(self.name))
        printer("Num Tels : {}".format(self.num_tels))
        printer("Footprint: {:.2f}".format(self.footprint))
        printer("")
        printer("                TYPE  Num IDmin  IDmax")
        printer("=====================================")
        for teltype, tels in teltypes.items():
            printer("{:>20s} {:4d} {:4d} ..{:4d}".format(teltype,
                                                         len(tels),
                                                         min(tels),
                                                         max(tels)))

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
    def pos_x(self):
        """ telescope x position as array """
        warnings.warn("SubarrayDescription.pos_x is deprecated. Use "
                      "tel_coords.x")
        return self.tel_coords.x

    @property
    def pos_y(self):
        """ telescope y positions as an array"""
        warnings.warn("SubarrayDescription.pos_y is deprecated. Use "
                      "tel_coords.y")
        return self.tel_coords.y

    @property
    def pos_z(self):
        """ telescope y positions as an array"""
        warnings.warn("SubarrayDescription.pos_z is deprecated. Use "
                      "tel_coords.z")
        return self.tel_coords.z

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

            ids = [x for x in self.tels]
            descs = [str(x) for x in self.tels.values()]
            mirror_types = [x.optics.mirror_type for x in self.tels.values()]
            tel_types = [x.optics.tel_type for x in self.tels.values()]
            tel_subtypes = [x.optics.tel_subtype for x in self.tels.values()]
            cam_types = [x.camera.cam_id for x in self.tels.values()]
            tel_coords = self.tel_coords

            tab = Table(dict(tel_id=np.array(ids, dtype=np.short),
                             tel_pos_x=tel_coords.x,
                             tel_pos_y=tel_coords.y,
                             tel_pos_z=tel_coords.z,
                             tel_type=tel_types,
                             tel_subtype=tel_subtypes,
                             mirror_type=mirror_types,
                             camera_type=cam_types,
                             tel_description=descs))

        elif kind == 'optics':

            optics_ids = {x.optics.identifier for x in self.tels.values()}
            optics_list = []

            # get one example of each OpticsDescription
            for oid in optics_ids:
                optics_list.append(next(x.optics for x in self.tels.values() if
                                        x.optics.identifier == oid))

            cols = {
                'tel_description': [str(x) for x in optics_list],
                'tel_type': [x.tel_type for x in optics_list],
                'tel_subtype': [x.tel_subtype for x in optics_list],
                'mirror_area': np.array([x.mirror_area.to('m2').value for x
                                         in optics_list]) * u.m ** 2,
                'mirror_type': [x.mirror_type for x in optics_list],
                'num_mirror_tiles': [x.num_mirror_tiles for x in optics_list],
                'equivalent_focal_length': [x.equivalent_focal_length.to('m')
                                            for x in optics_list] * u.m,
            }

            tab = Table(cols)

        else:
            raise ValueError("Table type '{}' not known".format(kind))

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

        types = {str(tel) for tel in self.tels.values()}
        tab = self.to_table()

        plt.figure(figsize=(8, 8))

        with quantity_support():
            for teltype in types:
                tels = tab[tab['tel_description'] == teltype]['tel_id']
                sub = self.select_subarray(teltype, tels)
                radius = np.array([np.sqrt(tel.optics.mirror_area / np.pi).value
                                   for tel in sub.tels.values()])

                plt.scatter(sub.pos_x, sub.pos_y, s=radius * 8, alpha=0.5,
                            label=teltype)

            plt.legend(loc='best')
            plt.title(self.name)
            plt.tight_layout()

    @property
    def telescope_types(self):
        """ list of telescope types in the array"""
        tel_types = {str(tt) for tt in self.tel.values()}
        return list(tel_types)

    @property
    def camera_types(self):
        """ list of camera types in the array """
        cam_types = {str(tt.camera) for tt in self.tel.values()}
        return list(cam_types)

    @property
    def optics_types(self):
        """ list of optics types in the array """
        cam_types = {str(tt.optics) for tt in self.tel.values()}
        return list(cam_types)

    def get_tel_ids_for_type(self, tel_type):
        """
        return list of tel_ids that have the given tel_type

        Parameters
        ----------
        tel_type: str
           telescope type string (e.g. 'MST:NectarCam')

        """
        return [id for id, descr in self.tels.items() if str(descr) == tel_type]
