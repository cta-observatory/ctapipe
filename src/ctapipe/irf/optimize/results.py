from typing import Sequence

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.table import QTable, Table

from ...core import QualityQuery
from ...core.traits import Path
from ...irf import ResultValidRange


class OptimizationResult:
    """Result of an optimization of G/H and theta cuts or only G/H cuts."""

    def __init__(
        self,
        valid_energy_min: u.Quantity,
        valid_energy_max: u.Quantity,
        valid_offset_min: u.Quantity,
        valid_offset_max: u.Quantity,
        gh_cuts: QTable,
        clf_prefix: str,
        spatial_selection_table: QTable | None = None,
        quality_query: QualityQuery | Sequence | None = None,
    ) -> None:
        if quality_query:
            if isinstance(quality_query, QualityQuery):
                if len(quality_query.quality_criteria) == 0:
                    quality_query.quality_criteria = [
                        (" ", " ")
                    ]  # Ensures table serialises properly

                self.quality_query = quality_query
            elif isinstance(quality_query, list):
                self.quality_query = QualityQuery(quality_criteria=quality_query)
            else:
                self.quality_query = QualityQuery(quality_criteria=list(quality_query))
        else:
            self.quality_query = QualityQuery(quality_criteria=[(" ", " ")])

        self.valid_energy = ResultValidRange(min=valid_energy_min, max=valid_energy_max)
        self.valid_offset = ResultValidRange(min=valid_offset_min, max=valid_offset_max)
        self.gh_cuts = gh_cuts
        self.clf_prefix = clf_prefix
        self.spatial_selection_table = spatial_selection_table

    def __repr__(self):
        if self.spatial_selection_table is not None:
            return (
                f"<OptimizationResult with {len(self.gh_cuts)} G/H bins "
                f"and {len(self.spatial_selection_table)} theta bins valid "
                f"between {self.valid_offset.min} to {self.valid_offset.max} "
                f"and {self.valid_energy.min} to {self.valid_energy.max} "
                f"with {len(self.quality_query.quality_criteria)} quality criteria>"
            )
        else:
            return (
                f"<OptimizationResult with {len(self.gh_cuts)} G/H bins valid "
                f"between {self.valid_offset.min} to {self.valid_offset.max} "
                f"and {self.valid_energy.min} to {self.valid_energy.max} "
                f"with {len(self.quality_query.quality_criteria)} quality criteria>"
            )

    def write(self, output_name: Path | str, overwrite: bool = False) -> None:
        """Write an ``OptimizationResult`` to a file in FITS format."""

        cut_expr_tab = Table(
            rows=self.quality_query.quality_criteria,
            names=["name", "cut_expr"],
            dtype=[np.str_, np.str_],
        )
        cut_expr_tab.meta["EXTNAME"] = "QUALITY_CUTS_EXPR"

        self.gh_cuts.meta["EXTNAME"] = "GH_CUTS"
        self.gh_cuts.meta["CLFNAME"] = self.clf_prefix

        energy_lim_tab = QTable(
            rows=[[self.valid_energy.min, self.valid_energy.max]],
            names=["energy_min", "energy_max"],
        )
        energy_lim_tab.meta["EXTNAME"] = "VALID_ENERGY"

        offset_lim_tab = QTable(
            rows=[[self.valid_offset.min, self.valid_offset.max]],
            names=["offset_min", "offset_max"],
        )
        offset_lim_tab.meta["EXTNAME"] = "VALID_OFFSET"

        results = [cut_expr_tab, self.gh_cuts, energy_lim_tab, offset_lim_tab]

        if self.spatial_selection_table is not None:
            self.spatial_selection_table.meta["EXTNAME"] = "RAD_MAX"
            results.append(self.spatial_selection_table)

        # Overwrite if needed and allowed
        results[0].write(output_name, format="fits", overwrite=overwrite)

        for table in results[1:]:
            table.write(output_name, format="fits", append=True)

    @classmethod
    def read(cls, file_name):
        """Read an ``OptimizationResult`` from a file in FITS format."""

        with fits.open(file_name) as hdul:
            cut_expr_tab = Table.read(hdul[1])
            cut_expr_lst = [(name, expr) for name, expr in cut_expr_tab.iterrows()]
            if (" ", " ") in cut_expr_lst:
                cut_expr_lst.remove((" ", " "))

            quality_query = QualityQuery(quality_criteria=cut_expr_lst)
            gh_cuts = QTable.read(hdul[2])
            valid_energy = QTable.read(hdul[3])
            valid_offset = QTable.read(hdul[4])
            spatial_selection_table = QTable.read(hdul[5]) if len(hdul) > 5 else None

        return cls(
            quality_query=quality_query,
            valid_energy_min=valid_energy["energy_min"],
            valid_energy_max=valid_energy["energy_max"],
            valid_offset_min=valid_offset["offset_min"],
            valid_offset_max=valid_offset["offset_max"],
            gh_cuts=gh_cuts,
            clf_prefix=gh_cuts.meta["CLFNAME"],
            spatial_selection_table=spatial_selection_table,
        )
