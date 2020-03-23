# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Utilities for reading or working with Camera geometry files
"""
import logging

import numpy as np
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.table import Table
from astropy.utils import lazyproperty
from scipy.spatial import cKDTree as KDTree
from scipy.sparse import lil_matrix, csr_matrix

from ctapipe.utils import get_table_dataset, find_all_matching_datasets
from ctapipe.coordinates import CameraFrame


__all__ = ['CameraReadout']

logger = logging.getLogger(__name__)


class CameraReadout:
    """
    Stores properties related to the readout of a Cherenkov Camera

    Parameters
    ----------
    self: type
        description
    cam_id: camera id name or number
        camera identification string
    sampling_rate
    reference_pulse_shape
    reference_pulse_step
    """

    def __init__(self, cam_id, sampling_rate, reference_pulse_shape, reference_pulse_step):
        self.cam_id = cam_id
        self.sampling_rate = sampling_rate
        self.reference_pulse_shape = reference_pulse_shape
        self.reference_pulse_step = reference_pulse_step

    def __eq__(self, other):
        if self.cam_id != other.cam_id:
            return False

        if self.sampling_rate != other.sampling_rate:
            return False

        if (self.reference_pulse_shape != other.reference_pulse_shape).all():
            return False

        if self.reference_pulse_step != other.reference_pulse_step:
            return False

        return True

    def __hash__(self):
        return hash((
            self.cam_id,
            self.sampling_rate.to_value(u.GHz),
            self.reference_pulse_shape.size,
            self.reference_pulse_step.to_value(u.ns),
        ))

    def __len__(self):
        return self.reference_pulse_shape.size

    @classmethod
    def from_name(cls, camera_id='NectarCam', version=None):
        """
        Construct a CameraReadout using the name of the camera and array.

        This expects that there is a resource in the `ctapipe_resources` module
        called "[array]-[camera].camreadout.fits.gz" or "[array]-[camera]-[
        version].camgeom.fits.gz"

        Parameters
        ----------
        camera_id: str
           name of camera (e.g. 'NectarCam', 'LSTCam', 'GCT', 'SST-1M')
        version:
           camera version id (currently unused)

        Returns
        -------
        new CameraReadout
        """

        if version is None:
            verstr = ''
        else:
            verstr = f"-{version:03d}"

        tabname = "{camera_id}{verstr}.camreadout".format(camera_id=camera_id,
                                                       verstr=verstr)
        table = get_table_dataset(tabname, role='dl0.tel.svc.camera')
        return CameraReadout.from_table(table)

    def to_table(self):
        """ convert this to an `astropy.table.Table` """
        return Table([self.reference_pulse_shape],
                     names=['reference_pulse_shape'],
                     meta=dict(TAB_TYPE='ctapipe.instrument.CameraReadout',
                               TAB_VER='1.0',
                               CAM_ID=self.cam_id,
                               SAMPFREQ=self.sampling_rate.to_value(u.GHz),
                               REF_STEP=self.reference_pulse_step.to_value(u.ns),
                               ))

    @classmethod
    def from_table(cls, url_or_table, **kwargs):
        """
        Load a CameraReadout from an `astropy.table.Table` instance or a
        file that is readable by `astropy.table.Table.read()`

        Parameters
        ----------
        url_or_table: string or astropy.table.Table
            either input filename/url or a Table instance
        kwargs: extra keyword arguments
            extra arguments passed to `astropy.table.read()`, depending on
            file type (e.g. format, hdu, path)


        """
        tab = url_or_table
        if not isinstance(url_or_table, Table):
            tab = Table.read(url_or_table, **kwargs)

        return cls(
            cam_id=tab.meta.get('CAM_ID', 'Unknown'),
            sampling_rate=u.Quantity(tab.meta["SAMPFREQ"], u.GHz),
            reference_pulse_shape=tab['reference_pulse_shape'],
            reference_pulse_step=u.Quantity(tab.meta['REF_STEP'], u.ns),
        )

    def __repr__(self):
        return (
            "CameraReadout(cam_id='{cam_id}', sampling_rate='{sampling_rate}', "
            "reference_pulse_step={reference_pulse_step})"
        ).format(
            cam_id=self.cam_id,
            sampling_rate=self.sampling_rate,
            reference_pulse_step=self.reference_pulse_step,
        )

    def __str__(self):
        return self.cam_id
