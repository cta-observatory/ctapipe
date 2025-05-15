from astropy.table import Table

from ...core import QualityQuery
from ...core.traits import List, Tuple, Unicode


class EventQualitySelection(QualityQuery):
    """
    Event pre-selection quality criteria for IRF and DL3 computation with different defaults.
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

    def calculate_selection(self, events: Table):
        """
        Add the selection columns to the events, will only compute quality selection

        Parameters
        ----------
        events: Table
            The table containing the events on which selection need to be applied

        Returns
        -------
        Table
            events with selection columns added.
        """
        return self.calculate_quality_selection(events)

    def calculate_quality_selection(self, events: Table):
        """
        Add the selection columns to the events, will only compute quality selection

        Parameters
        ----------
        events: Table
            The table containing the events on which selection need to be applied

        Returns
        -------
        Table
            events with selection columns added.
        """
        events["selected_quality"] = self.get_table_mask(events)
        events["selected"] = events["selected_quality"]
        return events
