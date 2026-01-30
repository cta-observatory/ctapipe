#!/usr/bin/env python3

"""
Components for computing event weights.

`~ctapipe.irf.EventWeighter` is a base Component with implementations of spectral event weighting:

* `~ctapipe.irf.SimpleEventWeighter`: spectral weighting for the full FOV
* `~ctapipe.irf.RadialEventWeighter`: spectral weighing in radial bins in the  FOV

They operate on a pre-processed table of DL2 information, where you specify the
energy and FOV coordinate columns to use. The interface is as follows:

.. code-block:: python

  from astropy.table import QTable

  from ctapipe.irf import RadialEventWeighter  spectrum_from_name

  table = QTable(
      dict(
          true_energy=[1.0, 2.0, 0.5, 0.2] * u.TeV,
          true_fov_offset=[0.1, 1.2, 2.2, 3.2] * u.deg,
      )
  )

  # the source spectrum can be anything, but is usually loaded
  # with ctapipe.irf.spectrum_from_simulation_config() to
  # weight simulations. Here we will just use the electron
  # spectrum as a demo:
  source_spectrum = spectrum_from_name("IRFDOC_ELECTRON_SPECTRUM")

  weighter = RadialEventWeighter(
      source_spectrum=source_spectrum
      target_spectrum_name="CRAB_HEGRA",
      fov_offset_max=5.0*u.deg,
      fov_offset_n_bins=5,
  )

  table_with_weights = weighter(table)
  print(table_with_weights)


::

  true_energy true_fov_offset       weight       fov_offset_bin
      TeV           deg
  ----------- --------------- ------------------ --------------
          1.0             0.1 12.399508446433442              1
          2.0             1.2  7.247055871572714              2
          0.5             2.2 1.4149219056909543              3
          0.2             3.2 0.4812883464910382              4


"""

from abc import abstractmethod
from collections.abc import Callable
from typing import override

import numpy as np
from astropy import units as u
from astropy.table import QTable, Table
from pyirf.binning import OVERFLOW_INDEX, UNDERFLOW_INDEX, calculate_bin_indices
from pyirf.spectral import (
    calculate_event_weights,
)

from ..core import Component, traits
from ..core.feature_generator import shallow_copy_table
from .binning import DefaultFoVOffsetBins
from .spectra import Spectra, spectrum_from_name

__all__ = [
    "EventWeighter",
    "SimpleEventWeighter",
    "RadialEventWeighter",
]


class EventWeighter(Component):
    """Compute weights to go from source to target spectra."""

    target_spectrum_name = traits.UseEnum(
        Spectra,
        default_value=Spectra.CRAB_HEGRA,
        help="Pre-defined source spectrum to reweight to.",
    ).tag(config=True)

    is_diffuse = traits.Bool(
        default_value=True, help="If True, assume the source is diffuse."
    ).tag(config=True)

    energy_column = traits.Unicode(
        help="name of energy column", default_value="true_energy"
    ).tag(config=True)
    weight_column = traits.Unicode(
        help="name of output weight column", default_value="weight"
    ).tag(config=True)

    def __init__(
        self,
        source_spectrum: Callable[[u.Quantity], u.Quantity],
        config=None,
        parent=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        source_spectrum: Callable
            initial spectrum of the events to be processed.
        """
        super().__init__(config=config, parent=parent, **kwargs)
        self.source_spectrum = source_spectrum
        self.target_spectrum = spectrum_from_name(self.target_spectrum_name)

    @abstractmethod
    def _compute_weights(self, events_table: QTable):
        raise NotImplementedError(
            f"{self.__class__.__name__} weighting is not implemented"
        )

    def __call__(self, events_table: Table | QTable) -> QTable:
        """Returns shallow copy of input table with a ``weight`` column added"""

        table = shallow_copy_table(events_table, output_cls=QTable)
        self._compute_weights(table)
        return table


class SimpleEventWeighter(EventWeighter):
    """Weights all events spectrally with no spatial binning.

    Calling this class adds a column to the output table with the event-wise
    spectral weights, with column name ``weight_column``.
    """

    fov_offset_max = traits.AstroQuantity(
        help="upper bound of spatial integral applied to source function",
        default_value=u.Quantity(10, u.deg),
        physical_type=u.physical.angle,
    ).tag(config=True)

    @override
    def _compute_weights(self, events_table: QTable):
        energy = events_table[self.energy_column]
        source_spectrum = self.source_spectrum
        if self.is_diffuse:
            source_spectrum = source_spectrum.integrate_cone(
                0 * u.deg, self.fov_offset_max
            )
        weights = calculate_event_weights(
            energy,
            target_spectrum=self.target_spectrum,
            simulated_spectrum=source_spectrum,
        )
        events_table[self.weight_column] = weights


class RadialEventWeighter(EventWeighter, DefaultFoVOffsetBins):
    """
    Weights in radial (FOV) offset bins in the `~ctapipe.coordinates.NominalFrame`.

    Calling this class adds a column to the output table with the event-wise
    spectral-spatial weights, with column name ``weight_column``. This
    implementation additionally adds the column
    ``output_table["fov_offset_bin"]`` and
    ``output_table["fov_offset_is_valid"]`` following the conventions of
    ``pyirf.binning.calculate_bin_indices``, and the list of offset bin edges in
    ``output_table.meta["OFFSBINS"]``
    """

    fov_offset_column = traits.Unicode(
        help="name of FOV radial offset column", default_value="true_fov_offset"
    ).tag(config=True)

    @override
    def _compute_weights(self, events_table: QTable):
        offset_bins = self.fov_offset_bins
        offset = events_table[self.fov_offset_column].to(offset_bins.unit)
        energy = events_table[self.energy_column]
        weights = np.zeros_like(energy.value)

        # note that the bin i from digitize starts at 1 and means:
        #    offset_bins[i-1] <= offset < offset_bins[ii])
        r_bin, r_valid = calculate_bin_indices(offset, offset_bins)

        for ii in range(len(offset_bins) - 1):
            self.log.debug(
                f"bin {ii} offset=[{offset_bins[ii]}, {offset_bins[ii + 1]})"
            )
            mask = r_bin == ii
            weights[mask] = calculate_event_weights(
                true_energy=energy[mask],
                target_spectrum=self.target_spectrum,
                simulated_spectrum=self.source_spectrum.integrate_cone(
                    offset_bins[ii], offset_bins[ii + 1]
                ),
            )

        events_table[self.weight_column] = weights
        events_table["fov_offset_bin"] = r_bin
        events_table["fov_offset_is_valid"] = r_valid

        events_table.columns["fov_offset_bin"].description = (
            "Bin i defined as offset[i-1] <= fov_offset < offset[i]. "
            "Where offset is `OFFSBINS` array found this table's metadata."
        )

        events_table.columns[
            "fov_offset_is_valid"
        ].description = "True if event's offset was inside the binning range."

        events_table.meta["OFFSBINS"] = list(offset_bins.to_value("deg"))
        events_table.meta["BINOVER"] = OVERFLOW_INDEX
        events_table.meta["BINUNDR"] = UNDERFLOW_INDEX
