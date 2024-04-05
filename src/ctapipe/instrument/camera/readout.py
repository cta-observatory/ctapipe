# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Utilities for reading or working with Camera geometry files
"""
import logging

import numpy as np
from astropy import units as u
from astropy.table import Table

from ctapipe.utils import get_table_dataset

from ..warnings import warn_from_name

__all__ = ["CameraReadout"]

logger = logging.getLogger(__name__)


def parse_dotted_version(version):
    return tuple(map(int, version.split(".")))


class CameraReadout:
    """Stores properties related to the readout of a Cherenkov Camera."""

    CURRENT_TAB_VERSION = "3.0"
    SUPPORTED_TAB_VERSIONS = {"3.0"}

    __slots__ = (
        "name",
        "sampling_rate",
        "reference_pulse_shape",
        "reference_pulse_sample_width",
        "n_channels",
        "n_pixels",
        "n_samples",
        "n_samples_long",
    )

    def __init__(
        self,
        name,
        sampling_rate,
        reference_pulse_shape,
        reference_pulse_sample_width,
        n_channels,
        n_pixels,
        n_samples,
        n_samples_long=None,
    ):
        """Stores properties related to the readout of a Cherenkov Camera.

        Parameters
        ----------
        name: str
             Camera name (e.g. NectarCam, LSTCam, ...)
        sampling_rate : u.Quantity[frequency]
            Sampling rate of the waveform
        reference_pulse_shape : ndarray
            Expected pulse shape for a signal in the waveform. 2 dimensional,
            first dimension is gain channel.
        reference_pulse_sample_width : u.Quantity[time]
            The amount of time corresponding to each sample in the 2nd
            dimension of reference_pulse_shape
        n_channels : int
            Number of gain channels
        n_pixels : int
            Number of pixels
        n_samples : int
            Number of waveform samples for normal events
        n_samples_long : int or None
            Number of waveform samples for long events. Not all cameras
            have long event types. Leave None if camera does not support long
            events.
        """
        self.name = name
        self.sampling_rate = sampling_rate
        self.reference_pulse_shape = reference_pulse_shape
        self.reference_pulse_sample_width = reference_pulse_sample_width
        self.n_channels = n_channels
        self.n_pixels = n_pixels
        self.n_samples = n_samples
        self.n_samples_long = n_samples_long

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        return (
            self.n_pixels == other.n_pixels
            and self.n_samples == other.n_samples
            and self.n_samples_long == other.n_samples_long
            and self.name == other.name
            and u.isclose(self.sampling_rate, other.sampling_rate)
            and u.isclose(
                self.reference_pulse_sample_width, other.reference_pulse_sample_width
            )
            and np.allclose(self.reference_pulse_shape, other.reference_pulse_shape)
        )

    def __hash__(self):
        return hash(
            (
                self.name,
                round(self.sampling_rate.to_value(u.GHz), 3),
                self.reference_pulse_shape.size,
                self.n_channels,
                self.n_pixels,
                self.n_samples,
                self.n_samples_long,
                round(self.reference_pulse_sample_width.to_value(u.ns), 2),
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
    def from_name(cls, name="NectarCam", version=None):
        """Construct a CameraReadout using the name of the camera and array.

        This expects that there is a resource accessible ``ctapipe_resources``
        via `~ctapipe.utils.get_table_dataset` called
        ``"[array]-[camera].camreadout.fits.gz"`` or
        ``"[array]-[camera]-[version].camgeom.fits.gz"``.

        Parameters
        ----------
        name: str
             Camera name (e.g. NectarCam, LSTCam, ...)
        version:
           camera version id (currently unused)

        Returns
        -------
        new CameraReadout
        """
        warn_from_name()

        if version is None:
            verstr = ""
        else:
            verstr = f"-{version:03d}"

        tabname = "{name}{verstr}.camreadout".format(name=name, verstr=verstr)
        table = get_table_dataset(tabname, role="dl0.tel.svc.camera")
        return CameraReadout.from_table(table)

    def to_table(self):
        """Convert this to an `astropy.table.Table`."""
        n_channels = self.n_channels
        columns = [
            *[self.reference_pulse_shape[i] for i in range(n_channels)],
            self.reference_pulse_sample_time,
        ]
        names = [
            *[f"reference_pulse_shape_channel{i}" for i in range(n_channels)],
            "reference_pulse_sample_time",
        ]
        meta = dict(
            TAB_TYPE="ctapipe.instrument.CameraReadout",
            TAB_VER=self.CURRENT_TAB_VERSION,
            CAM_ID=self.name,
            NCHAN=n_channels,
            NPIXELS=self.n_pixels,
            NSAMPLES=self.n_samples,
            SAMPFREQ=self.sampling_rate.to_value(u.GHz),
            REFWIDTH=self.reference_pulse_sample_width.to_value(u.ns),
        )

        if self.n_samples_long is not None:
            meta["NSAMPLNG"] = self.n_samples_long

        return Table(
            columns,
            names=names,
            meta=meta,
        )

    @classmethod
    def from_table(cls, url_or_table, **kwargs):
        """Load a CameraReadout from an `astropy.table.Table` instance or a
        file that is readable by `astropy.table.Table.read`.

        Parameters
        ----------
        url_or_table: string or astropy.table.Table
            either input filename/url or a Table instance
        kwargs: extra keyword arguments
            extra arguments passed to `astropy.table.Table.read`, depending on
            file type (e.g. format, hdu, path)

        """
        tab = url_or_table
        if not isinstance(url_or_table, Table):
            tab = Table.read(url_or_table, **kwargs)

        version = tab.meta.get("TAB_VER", "")
        if version not in cls.SUPPORTED_TAB_VERSIONS:
            raise OSError(
                f"CameraReadout table has unsupported version: {version},"
                f" supported are: {cls.SUPPORTED_TAB_VERSIONS}."
            )

        name = tab.meta.get("CAM_ID", "Unknown")
        n_channels = tab.meta["NCHAN"]
        sampling_rate = u.Quantity(tab.meta["SAMPFREQ"], u.GHz)

        reference_pulse_sample_width = u.Quantity(tab.meta["REFWIDTH"], u.ns)
        reference_pulse_shape = np.array(
            [tab[f"reference_pulse_shape_channel{i}"] for i in range(n_channels)]
        )

        return cls(
            name=name,
            sampling_rate=sampling_rate,
            reference_pulse_shape=reference_pulse_shape,
            reference_pulse_sample_width=reference_pulse_sample_width,
            n_channels=tab.meta["NCHAN"],
            n_pixels=tab.meta["NPIXELS"],
            n_samples=tab.meta["NSAMPLES"],
            n_samples_long=tab.meta.get("NSAMPLNG"),
        )

    def __repr__(self):
        return (
            f"CameraReadout(name={self.name!r}"
            f", sampling_rate={self.sampling_rate}"
            f", n_channels={self.n_channels}"
            f", n_pixels={self.n_pixels}"
            f", n_samples={self.n_samples}"
            f", n_samples_long={self.n_samples_long}"
            ")"
        )

    def __str__(self):
        return self.name
