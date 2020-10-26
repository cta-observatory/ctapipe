# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Utilities for reading or working with Camera geometry files
"""
import logging

import numpy as np
from astropy import units as u
from astropy.table import Table
from scipy.stats import norm
from ctapipe.utils import get_table_dataset


__all__ = ["CameraReadout"]

logger = logging.getLogger(__name__)


class CameraReadout:
    def __init__(
        self,
        camera_name,
        sampling_rate,
        reference_pulse_shape,
        reference_pulse_sample_width,
    ):
        """Stores properties related to the readout of a Cherenkov Camera.

        Parameters
        ----------
        camera_name: str
             Camera name (e.g. NectarCam, LSTCam, ...)
        sampling_rate : u.Quantity[frequency]
            Sampling rate of the waveform
        reference_pulse_shape : ndarray
            Expected pulse shape for a signal in the waveform. 2 dimensional,
            first dimension is gain channel.
        reference_pulse_sample_width : u.Quantity[time]
            The amount of time corresponding to each sample in the 2nd
            dimension of reference_pulse_shape

        """
        self.camera_name = camera_name
        self.sampling_rate = sampling_rate
        self.reference_pulse_shape = reference_pulse_shape
        self.reference_pulse_sample_width = reference_pulse_sample_width

    def __eq__(self, other):
        if self.camera_name != other.camera_name:
            return False

        if self.sampling_rate != other.sampling_rate:
            return False

        if (self.reference_pulse_shape != other.reference_pulse_shape).all():
            return False

        if self.reference_pulse_sample_width != other.reference_pulse_sample_width:
            return False

        return True

    def __hash__(self):
        return hash(
            (
                self.camera_name,
                self.sampling_rate.to_value(u.GHz),
                self.reference_pulse_shape.size,
                self.reference_pulse_sample_width.to_value(u.ns),
            )
        )

    def __len__(self):
        return self.reference_pulse_shape.size

    @property
    def reference_pulse_sample_time(self):
        """
        Time axis for the reference pulse
        """
        _, n_samples = self.reference_pulse_shape.shape
        sample_width_ns = self.reference_pulse_sample_width.to_value(u.ns)
        pulse_max_sample = n_samples * sample_width_ns
        sample_time = np.arange(0, pulse_max_sample, sample_width_ns)
        return u.Quantity(sample_time, u.ns)

    @classmethod
    def from_name(cls, camera_name="NectarCam", version=None):
        """Construct a CameraReadout using the name of the camera and array.

        This expects that there is a resource in the `ctapipe_resources` module
        called "[array]-[camera].camreadout.fits.gz" or "[array]-[camera]-[
        version].camgeom.fits.gz".

        Parameters
        ----------
        camera_name: str
             Camera name (e.g. NectarCam, LSTCam, ...)
        version:
           camera version id (currently unused)

        Returns
        -------
        new CameraReadout

        """

        if version is None:
            verstr = ""
        else:
            verstr = f"-{version:03d}"

        try:
            tabname = "{camera_name}{verstr}.camreadout".format(
                camera_name=camera_name, verstr=verstr
            )
            table = get_table_dataset(tabname, role="dl0.tel.svc.camera")
            return CameraReadout.from_table(table)
        except FileNotFoundError:
            # TODO: remove case when files have been generated
            logger.warning(
                f"Resorting to default CameraReadout,"
                f" File does not exist: ({tabname})"
            )
            reference_pulse_shape = np.array([norm.pdf(np.arange(96), 48, 6)])
            return cls(
                camera_name=camera_name,
                sampling_rate=u.Quantity(1, u.GHz),
                reference_pulse_shape=reference_pulse_shape,
                reference_pulse_sample_width=u.Quantity(1, u.ns),
            )

    def to_table(self):
        """Convert this to an `astropy.table.Table`."""
        n_channels = len(self.reference_pulse_shape)
        tables = [
            *[self.reference_pulse_shape[i] for i in range(n_channels)],
            self.reference_pulse_sample_time,
        ]
        names = [
            *[f"reference_pulse_shape_channel{i}" for i in range(n_channels)],
            "reference_pulse_sample_time",
        ]

        return Table(
            tables,
            names=names,
            meta=dict(
                TAB_TYPE="ctapipe.instrument.CameraReadout",
                TAB_VER="1.0",
                CAM_ID=self.camera_name,
                NCHAN=n_channels,
                SAMPFREQ=self.sampling_rate.to_value(u.GHz),
                REF_WIDTH=self.reference_pulse_sample_width.to_value(u.ns),
            ),
        )

    @classmethod
    def from_table(cls, url_or_table, **kwargs):
        """Load a CameraReadout from an `astropy.table.Table` instance or a
        file that is readable by `astropy.table.Table.read()`.

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

        camera_name = tab.meta.get("CAM_ID", "Unknown")
        n_channels = tab.meta["NCHAN"]
        sampling_rate = u.Quantity(tab.meta["SAMPFREQ"], u.GHz)
        reference_pulse_sample_width = u.Quantity(tab.meta["REF_WIDTH"], u.ns)
        reference_pulse_shape = np.array(
            [tab[f"reference_pulse_shape_channel{i}"] for i in range(n_channels)]
        )

        return cls(
            camera_name=camera_name,
            sampling_rate=sampling_rate,
            reference_pulse_shape=reference_pulse_shape,
            reference_pulse_sample_width=reference_pulse_sample_width,
        )

    def __repr__(self):
        return (
            "CameraReadout(camera_name='{camera_name}', sampling_rate='{sampling_rate}', "
            "reference_pulse_sample_width={reference_pulse_sample_width})"
        ).format(
            camera_name=self.camera_name,
            sampling_rate=self.sampling_rate,
            reference_pulse_sample_width=self.reference_pulse_sample_width,
        )

    def __str__(self):
        return self.camera_name
