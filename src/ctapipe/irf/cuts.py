import operator

from astropy.table import Table
from pyirf.cuts import evaluate_binned_cut

from ctapipe.irf import OptimizationResult


def calculate_selections(
    events: Table, cuts: OptimizationResult, apply_spatial_selection: bool = False
) -> Table:
    """
    Add the selection columns to the events

    Parameters
    ----------
    events: Table
        The table containing the events on which selection need to be applied
    cuts: OptimizationResult
        The cuts that need to be applied on the events
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
        cuts.gh_cuts,
        operator.ge,
    )

    if apply_spatial_selection:
        events["selected_theta"] = evaluate_binned_cut(
            events["theta"],
            events["reco_energy"],
            cuts.spatial_selection_table,
            operator.le,
        )
        events["selected"] = events["selected_theta"] & events["selected_gh"]
    else:
        events["selected"] = events["gammas"]["selected_gh"]

    return events
