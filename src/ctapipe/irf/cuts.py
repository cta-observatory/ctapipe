import operator

from astropy.table import Table
from pyirf.cuts import evaluate_binned_cut

from ctapipe.core import QualityQuery
from ctapipe.core.traits import List, Path, Tuple, Unicode
from ctapipe.irf import OptimizationResult

__all__ = ["EventQualitySelection", "EventSelection"]


class EventQualitySelection(QualityQuery):
    """
    Event pre-selection quality criteria for IRF computation with different defaults.
    """

    quality_criteria = List(
        Tuple(Unicode(), Unicode()),
        default_value=[
            (
                "multiplicity 4",
                "np.count_nonzero(HillasReconstructor_telescopes,axis=1) >= 4",
            ),
            ("valid classifier", "RandomForestClassifier_is_valid"),
            ("valid geom reco", "HillasReconstructor_is_valid"),
            ("valid energy reco", "RandomForestRegressor_is_valid"),
        ],
        help=QualityQuery.quality_criteria.help,
    ).tag(config=True)

    def calculate_selection(self, events):
        return self.calculate_quality_selection(events)

    def calculate_quality_selection(self, events):
        events["selected_quality"] = self.get_table_mask(events)
        events["selected"] = events["selected_quality"]
        return events


class EventSelection(EventQualitySelection):
    """
    Event selection
    """

    cuts_file = Path(
        default_value=None,
        allow_none=False,
        directory_ok=False,
        help="Path to the cuts file to apply to the observation.",
    ).tag(config=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cuts = OptimizationResult.read(self.cuts_file)

    def calculate_selection(self, events):
        events = self.calculate_quality_selection(events)
        events = self.calculate_gamma_selection(events)
        events["selected"] = events["selected_quality"] & events["selected_gamma"]
        return events

    def calculate_gamma_selection(
        self, events: Table, apply_spatial_selection: bool = False
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
