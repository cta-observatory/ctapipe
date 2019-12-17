"""
Data Quality selection
"""

__all__ = ["Selector", "SelectionFunctionError"]

from collections import Callable

import numpy as np

from .component import Component
from .traits import Dict

class SelectionFunctionError(TypeError):
    pass

class Selector(Component):
    """
    Manages a set of user-configurable (at runtime or in a config file) selection
    criteria that operate on the same type of input. Each time it is called, it
    returns a boolean array of whether or not each criterion passed. It  also keeps
    track of the total number of times each criterium is passed, as well as a
    cumulative product of criterium (i.e. the criteria applied in-order)
    """

    selection_functions = Dict(
        help=(
            "dict of '<cut name>' : lambda function in string format to accept ("
            "select) a given data value.  E.g. `{'mycut': 'lambda x: x > 3'}` "
        )
    ).tag(config=True)

    def __init__(self, config=None, parent=None, **kwargs):
        super().__init__(config=config, parent=parent, **kwargs)

        # add a selection to count all entries and make it the first one
        selection_functions = {"TOTAL": "lambda x: True"}
        selection_functions.update(self.selection_functions)

        self.selection_functions = selection_functions  # update

        try:# generate real functions from the selection function strings
            self._selectors = {
                name: eval(func_str) for name, func_str in selection_functions.items()
            }
        except NameError as err:
            raise SelectionFunctionError("Couldn't evaluate one of the selection "
                                         "function strings")

        for name, func in self._selectors.items():
            if not isinstance(func, Callable):
                raise SelectionFunctionError(f"Selection criterion '{name}' is not a function")


        # arrays for recording overall statistics
        self._counts = np.zeros(len(self._selectors), dtype=np.int)
        self._cumulative_counts = np.zeros(len(self._selectors), dtype=np.int)

    def __len__(self):
        """ return number of events processed"""
        return self._counts[0]

    @property
    def criteria_names(self):
        """ list of names of criteria to be considered """
        return list(self._selectors.keys())

    @property
    def selection_function_strings(self):
        """ list of criteria function strings"""
        return list(self.selection_functions.values())

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
        result = np.array(list(map(lambda f: f(value), self._selectors.values())))
        self._counts += result.astype(int)
        self._cumulative_counts += result.cumprod()
        return result[1:]
