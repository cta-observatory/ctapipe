"""
Data Quality selection
"""

__all__ = ["QualityQuery", "QualityCriteriaError"]

from typing import Literal, Union

import numpy as np  # for use in selection functions
from astropy.table import Table

from .expression_engine import ExpressionEngine
from .telescope_component import (
    TelescopeComponent,
    TelescopeParameter,
    TelescopeParameterLookup,
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

    It is possible to use this without attaching a subarray (default is ``None``).
    """

    quality_criteria = List(
        Tuple(Unicode(), TelescopeParameter(Unicode())),
        help="List of tuples of ('query name', TelescopeParameter or expression) to accept"
        " (select) a given data value. Example: ``[('positive', 'x > 0')]`` or"
        " ``[('high_intensity', [('type', '*', 'hillas_intensity > 50'), ('type', 'SST*', 'hillas_intensity > 100')])]``"
        " You may use ``numpy`` as ``np`` and ``astropy.units`` as ``u``,"
        " but no other modules.",
    ).tag(config=True)

    def __init__(self, subarray=None, config=None, parent=None, **kwargs):
        super().__init__(subarray=subarray, config=config, parent=parent, **kwargs)

        self._compile()

    def _compile(self):
        self.engines = []
        self.criteria_names = []

        for name, criteria_list in self.quality_criteria:

            for i, crit in enumerate(criteria_list):
                command, arg, expr = crit
                par = (command, arg, ExpressionEngine(((name, expr),)))
                criteria_list[i] = par

            self.criteria_names.append(name)

            lookup = TelescopeParameterLookup(criteria_list)
            if self.subarray is not None:
                lookup.attach_subarray(self.subarray)
            self.engines.append((name, lookup))

        # arrays for recording overall statistics, add one for total count
        n = len(self.criteria_names) + 1
        self._counts = np.zeros(n, dtype=np.int64)
        self._cumulative_counts = np.zeros(n, dtype=np.int64)

    def to_table(self) -> Table:
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

    def __call__(
        self,
        key: Union[int, str, Literal[None]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Test that value passes all cuts

        Parameters
        ----------
        key : int, str, TelescopeDescription, or None
            Lookup parameter for the `ctapipe.core.telescope_component.TelescopeParameter`.
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
            expr = lookup[key]
            result[i] = next(expr(kwargs))

        self._counts += result.astype(int)
        self._cumulative_counts += result.cumprod()
        return result[1:]  # strip off TOTAL criterion, since redundant

    def get_table_mask(self, table: Table, key: Union[str, Literal[None]] = None):
        """
        Get a boolean mask for the entries that pass the quality checks.

        Parameters
        ----------
        table : `~astropy.table.Table`
            Table with columns matching the expressions used in the
            `QualityQuery.quality_criteria`.
        key : str or None
            Column name of the lookup parameter for the `ctapipe.core.telescope_component.TelescopeParameter`.

        Returns
        -------
        mask : np.ndarray[bool]
            Boolean mask of valid entries.
        """
        n_criteria = len(self.criteria_names) + 1
        tlen = len(table)

        if n_criteria == 1:
            self._counts += tlen
            self._cumulative_counts += tlen
            return np.ones(tlen, dtype=bool)

        result = np.ones((n_criteria, tlen), dtype=bool)

        if key is None:
            for i, (_, engine) in enumerate(self.engines, start=1):
                result[i] = next(engine[key](table))

        else:
            index_table = Table({key: table[key], "index": np.arange(tlen)})
            grouped = index_table.group_by(key)
            del index_table

            for i, (_, engine) in enumerate(self.engines, start=1):
                for group_keys, group_index in zip(grouped.groups.keys, grouped.groups):
                    _key = group_keys[key]
                    index = group_index["index"]
                    result[i][index] = next(engine[_key](table[index]))

        self._counts += np.count_nonzero(result, axis=1)
        self._cumulative_counts += np.count_nonzero(np.cumprod(result, axis=0), axis=1)
        return np.all(result, axis=0)
