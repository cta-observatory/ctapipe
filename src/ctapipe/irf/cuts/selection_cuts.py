import operator

from astropy.table import Table
from pyirf.cuts import evaluate_binned_cut

from ...core.traits import Path
from ..optimize.results import OptimizationResult
from .quality_cuts import EventQualitySelection

__all__ = ["EventSelection"]


class EventSelection(EventQualitySelection):
    """
    Event selection quality and gammaness criteria for IRF and DL3
    """

    cuts_file = Path(
        allow_none=False,
        directory_ok=False,
        exists=True,
        help="Path to the cuts file to apply to the observation.",
    ).tag(config=True)

    def __init__(self, config=None, parent=None, **kwargs):
        super().__init__(config=config, parent=parent, **kwargs)
        self.cuts = OptimizationResult.read(self.cuts_file)

    def calculate_selection(self, events: Table, apply_spatial_selection: bool = False):
        """
        Add the selection columns to the events

        Parameters
        ----------
        events: Table
            The table containing the events on which selection need to be applied
        apply_spatial_selection: bool
            True if the theta cuts should be applied

        Returns
        -------
        Table
            events with selection columns added.
        """
        events = self.calculate_quality_selection(events)
        events = self.calculate_gamma_selection(events, apply_spatial_selection)
        events["selected"] = events["selected_quality"] & events["selected_gamma"]
        return events

    def calculate_gamma_selection(
        self, events: Table, apply_spatial_selection: bool = False
    ) -> Table:
        """
        Add the selection columns to the events, will compute only gamma criteria

        Parameters
        ----------
        events: Table
            The table containing the events on which selection need to be applied
        apply_spatial_selection: bool
            True if the theta cuts should be applied

        Returns
        -------
        Table
            events with selection columns added.
        """

        events["selected_gh"] = evaluate_binned_cut(
            events["gh_score"],
            events["reco_energy"],
            self.cuts.gh_cuts,
            operator.ge,
        )

        if apply_spatial_selection:
            events["selected_theta"] = evaluate_binned_cut(
                events["theta"],
                events["reco_energy"],
                self.cuts.spatial_selection_table,
                operator.le,
            )
            events["selected_gamma"] = events["selected_theta"] & events["selected_gh"]
        else:
            events["selected_gamma"] = events["selected_gh"]

        events["selected"] = events["selected_gamma"]
        return events
