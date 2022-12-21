"""
Data Quality selection
"""

__all__ = ["QualityQuery", "QualityCriteriaError"]

import numpy as np  # for use in selection functions

from .component import Component
from .expression_engine import ExpressionEngine
from .traits import List, Tuple, Unicode


class QualityCriteriaError(TypeError):
    """Signal a problem with a user-defined selection criteria function"""


class QualityQuery(Component):
    """
    Manages a set of user-configurable (at runtime or in a config file) selection
    criteria that operate on the same type of input. Each time it is called, it
    returns a boolean array of whether or not each criterion passed. It  also keeps
    track of the total number of times each criterium is passed, as well as a
    cumulative product of criterium (i.e. the criteria applied in-order)
    """

    quality_criteria = List(
        Tuple(Unicode(), Unicode()),
        help=(
            "list of tuples of ('<description', 'expression string') to accept "
            "(select) a given data value.  E.g. ``[('mycut', 'x > 3'),]``. "
            "You may use ``numpy`` as ``np`` and ``astropy.units`` as ``u``,"
            " but no other modules."
        ),
    ).tag(config=True)

    def __init__(self, config=None, parent=None, **kwargs):
        super().__init__(config=config, parent=parent, **kwargs)

        # add a selection to count all entries and make it the first one
        self.criteria_names = [n for (n, _) in self.quality_criteria]
        self.expressions = [e for (_, e) in self.quality_criteria]

        self.engine = ExpressionEngine(self.quality_criteria)
        for _, expr in self.quality_criteria:
            if "lambda" in expr:
                raise ValueError(
                    "As of ctapipe 0.16, do not give lambda expressions"
                    " to QualityQuery. Directly give the expression."
                    " E.g. instead of `lambda p: p.hillas.width.value > 0`"
                    " use `parameters.hillas.width.value > 0`"
                )

        # arrays for recording overall statistics, add one for total count
        n = len(self.quality_criteria) + 1
        self._counts = np.zeros(n, dtype=np.int64)
        self._cumulative_counts = np.zeros(n, dtype=np.int64)

    def to_table(self, functions=False):
        """
        Return a tabular view of the latest quality summary

        The columns are
        - *criteria*: name of each criterion
        - *counts*: counts of each criterion independently
        - *cum_counts*: counts of cumulative application of each criterion in order

        Parameters
        ----------
        functions: bool:
            include the function string as a column

        Returns
        -------
        astropy.table.Table
        """
        from astropy.table import Table

        cols = {
            "criteria": ["TOTAL"] + self.criteria_names,
            "counts": self._counts,
            "cumulative_counts": self._cumulative_counts,
        }
        if functions:
            cols["func"] = ["True"] + self.expressions
        return Table(cols)

    def _repr_html_(self):
        """display nicely in Jupyter notebooks"""
        return self.to_table()._repr_html_()

    def __str__(self):
        """Print a formatted string representation of the entire table."""
        return str(self.to_table())

    def __call__(self, **kwargs) -> np.ndarray:
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
        result = np.ones(len(self.quality_criteria) + 1, dtype=bool)

        for i, res in enumerate(self.engine(kwargs), start=1):
            result[i] = res

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
        n_criteria = len(self.quality_criteria) + 1
        result = np.ones((n_criteria, len(table)), dtype=bool)

        for i, res in enumerate(self.engine(table), start=1):
            result[i] = res

        self._counts += np.count_nonzero(result, axis=1)
        self._cumulative_counts += np.count_nonzero(np.cumprod(result, axis=0), axis=1)
        return np.all(result, axis=0)
