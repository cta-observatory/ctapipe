"""
Data Quality selection
"""

__all__ = ["QualityQuery", "QualityCriteriaError"]

from collections.abc import Callable

import astropy.units as u  # for use in selection functions
import numpy as np  # for use in selection functions

from .component import Component
from .traits import List

# the following are what are allowed to be used
# in selection functions (passed to eval())
ALLOWED_GLOBALS = {
    "u": u,  # astropy units
    "np": np,  # numpy
}


class QualityCriteriaError(TypeError):
    """ Signal a problem with a user-defined selection criteria function"""

    pass


class QualityQuery(Component):
    """
    Manages a set of user-configurable (at runtime or in a config file) selection
    criteria that operate on the same type of input. Each time it is called, it
    returns a boolean array of whether or not each criterion passed. It  also keeps
    track of the total number of times each criterium is passed, as well as a
    cumulative product of criterium (i.e. the criteria applied in-order)
    """

    quality_criteria = List(
        help=(
            "list of tuples of ('<description', 'function string') to accept "
            "(select) a given data value.  E.g. `[('mycut', 'lambda x: x > 3'),]. "
            "You may use `numpy` as `np` and `astropy.units` as `u`, but no other"
            " modules."
        )
    ).tag(config=True)

    def __init__(self, config=None, parent=None, **kwargs):
        super().__init__(config=config, parent=parent, **kwargs)

        # add a selection to count all entries and make it the first one
        self.quality_criteria.insert(0, ("TOTAL", "lambda x: True"))
        self.criteria_names = []
        self.selection_function_strings = []
        self._selectors = []

        for name, func_str in self.quality_criteria:
            try:  # generate real functions from the selection function strings
                self.criteria_names.append(name)
                self.selection_function_strings.append(func_str)

                func = eval(func_str, ALLOWED_GLOBALS)
                if not isinstance(func, Callable):
                    raise QualityCriteriaError(
                        f"Selection criterion '{name}' cannot be evaluated because "
                        f" '{func_str}' is not a callable function"
                    )
                self._selectors.append(func)

            except NameError as err:
                # catch functions that cannot be defined. Note that this cannot check
                # that the function can run, that only happens the first time it's
                # called.
                raise QualityCriteriaError(
                    f"Couldn't evaluate selection function '{name}' -> '{func_str}' "
                    f"because: {err}"
                )

        # arrays for recording overall statistics
        self._counts = np.zeros(len(self._selectors), dtype=np.int)
        self._cumulative_counts = np.zeros(len(self._selectors), dtype=np.int)

    def __len__(self):
        """ return number of events processed"""
        return self._counts[0]

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
            "criteria": self.criteria_names,
            "counts": self._counts,
            "cumulative_counts": self._cumulative_counts,
        }
        if functions:
            cols["func"] = self.selection_function_strings
        return Table(cols)

    def _repr_html_(self):
        """display nicely in Jupyter notebooks"""
        return self.to_table()._repr_html_()

    def __call__(self, value) -> np.ndarray:
        """
        Test that value passes all cuts

        Parameters
        ----------
        value:
            the value to pass to each selection function

        Returns
        -------
        np.ndarray:
            array of booleans with results of each selection criterion in order
        """
        result = np.array(list(map(lambda f: f(value), self._selectors)))
        self._counts += result.astype(int)
        self._cumulative_counts += result.cumprod()
        return result[1:]  # strip off TOTAL criterion, since redundant
