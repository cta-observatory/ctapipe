"""
Data Quality selection
"""

__all__ = ["QualityQuery", "QualityCriteriaError"]

import numpy as np  # for use in selection functions
from astropy.table import Table

from .expression_engine import ExpressionEngine
from .telescope_component import (
    TelescopeComponent,
    TelescopeParameter,
    TelescopeParameterLookup,
    TelescopePatternList,
)
from .traits import List, Tuple, Unicode


class QualityCriteriaError(TypeError):
    """Signal a problem with a user-defined selection criteria function"""


class QualityQuery(TelescopeComponent):
    """
    Manages a set of user-configurable (at runtime or in a config file) selection
    criteria that operate on the same type of input. Each time it is called, it
    returns a boolean array of whether or not each criterion passed. It  also keeps
    track of the total number of times each criterium is passed, as well as a
    cumulative product of criterium (i.e. the criteria applied in-order)
    """

    quality_criteria = List(
        Tuple(Unicode(), TelescopeParameter(Unicode())),
        help="List of tuples of ('query name', TelescopeParameter or expression) to accept"
        " (select) a given data value. Example: ``[('positive', 'x > 0')]`` or"
        " ``[('high_intensity', [('type', '*', 'hillas_intensity > 50'), ('type', 'SST*', 'hillas_intensity > 100')])]``"
        " You may use ``numpy`` as ``np`` and ``astropy.units`` as ``u``,"
        " but no other modules.",
    ).tag(config=True)

    def __init__(self, subarray, config=None, parent=None, **kwargs):
        super().__init__(subarray=subarray, config=config, parent=parent, **kwargs)

        self._compile()

    def _compile(self):
        self.engines = []
        self.criteria_names = []

        for name, criteria_list in self.quality_criteria:

            if not isinstance(criteria_list, (tuple, list, TelescopePatternList)):
                criteria_list = TelescopePatternList([criteria_list])

            for i, crit in enumerate(criteria_list):
                command, arg, expr = crit
                par = (command, arg, ExpressionEngine(((name, expr),)))
                criteria_list[i] = par

            self.criteria_names.append(name)

            lookup = TelescopeParameterLookup(criteria_list)
            lookup.attach_subarray(self.subarray)
            self.engines.append((name, lookup))

        # arrays for recording overall statistics, add one for total count
        n = len(self.criteria_names) + 1
        self._counts = np.zeros(n, dtype=np.int64)
        self._cumulative_counts = np.zeros(n, dtype=np.int64)

    def to_table(self, functions=False) -> Table:
        """
        Return a tabular view of the latest quality summary

        The columns are
        - *criteria*: name of each criterion
        - *counts*: counts of each criterion independently
        - *cum_counts*: counts of cumulative application of each criterion in order

        Returns
        -------
        astropy.table.Table
        """

        cols = {
            "criteria": ["TOTAL"] + self.criteria_names,
            "counts": self._counts,
            "cumulative_counts": self._cumulative_counts,
        }
        return Table(cols)

    def _repr_html_(self):
        """display nicely in Jupyter notebooks"""
        return self.to_table()._repr_html_()

    def __str__(self):
        """Print a formatted string representation of the entire table."""
        return str(self.to_table())

    def __call__(self, tel_id, **kwargs) -> np.ndarray:
        """
        Test that value passes all cuts

        Parameters
        ----------
        **kwargs:
            Are passed as locals to evaluate the given expression

        Returns
        -------
        np.ndarray:
            array of booleans with results of each selection criterion in order
        """
        # add 1 for total
        result = np.ones(len(self.engines) + 1, dtype=bool)

        for i, (_, lookup) in enumerate(self.engines, start=1):
            expr = lookup[tel_id]
            result[i] = list(expr(kwargs))[0]

        self._counts += result.astype(int)
        self._cumulative_counts += result.cumprod()
        return result[1:]  # strip off TOTAL criterion, since redundant

    def get_table_mask(self, table):
        """
        Get a boolean mask for the entries that pass the quality checks.

        Parameters
        ----------
        table : `~astropy.table.Table`
            Table with columns matching the expressions used in the
            `QualityQuery.quality_criteria`.

        Returns
        -------
        mask : np.ndarray[bool]
            Boolean mask of valid entries.
        """
        n_criteria = len(self.criteria_names) + 1
        result = np.ones((n_criteria, len(table)), dtype=bool)

        for i, (_, engine) in enumerate(self.engines, start=1):
            for tel_id in np.unique(table["tel_id"]):
                tel_mask = table["tel_id"] == tel_id
                result[i][tel_mask] = list(engine[tel_id](table[tel_mask]))[0]

        self._counts += np.count_nonzero(result, axis=1)
        self._cumulative_counts += np.count_nonzero(np.cumprod(result, axis=0), axis=1)
        return np.all(result, axis=0)
