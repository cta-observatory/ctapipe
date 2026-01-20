"""
Generate Features.
"""

from collections import ChainMap

from astropy.table import QTable, Table

from .component import Component
from .expression_engine import ExpressionEngine
from .traits import List, Tuple, Unicode

__all__ = [
    "FeatureGenerator",
    "FeatureGeneratorException",
]


def _shallow_copy_table(table):
    """
    Make a shallow copy of the table.

    Data of the existing columns will be shared between shallow
    copies, but adding / removing columns won't be seen in
    the original table.
    """
    # automatically return Table or QTable depending on input
    return table.__class__({col: table[col] for col in table.colnames}, copy=False)


class FeatureGeneratorException(TypeError):
    """Signal a problem with a user-defined selection criteria function"""


class FeatureGenerator(Component):
    """
    Generate features for astropy.table.Table.

    Raises Exceptions in two cases:
    1. If a feature already exists in the table
    2. If a feature cannot be built with the given expression
    """

    features = List(
        Tuple(Unicode(), Unicode()),
        help=(
            "List of 2-Tuples of Strings: ('new_feature_name', 'expression to generate feature'). "
            "You can use ``numpy`` as ``np`` and ``astropy.units`` as ``u``. "
            "Several math functions are usable without the ``np``-prefix. "
            "Use ``feature.quantity.to_value(unit)`` to create features without units."
        ),
    ).tag(config=True)

    def __init__(self, config=None, parent=None, **kwargs):
        super().__init__(config=config, parent=parent, **kwargs)
        self.engine = ExpressionEngine(expressions=self.features)
        self._feature_names = [name for name, _ in self.features]

    def __call__(self, table: Table | QTable, **kwargs) -> Table:
        """
        Apply feature generation to the input table.

        This method returns a shallow copy of the input table with the
        new features added. Existing columns will share the underlying data,
        however the new columns won't be visible in the input table.

        Parameters
        ----------
        table: QTable | Table
            Input table. Internally a Table will be converted to a QTable so that
            unit propagation works, so expressions should only rely on properties of QTables.
        **kwargs:
            Other objects that should be available in expressions. For example,
            if a you pass ``subarray=subarray``, expressions can use that
            object. This can also be special functions like ``f=my_function``,
            which would allow an expression like ``"f(col1)"``.

        Returns
        -------
        QTable|Table:
            A new table with the same columns as the input, but with new columns
            for each feature. The returned class depends on what was passed in.
        """
        table_copy = _shallow_copy_table(QTable(table))
        lookup = ChainMap(table_copy, kwargs)

        for result, name in zip(self.engine(lookup), self._feature_names):
            if name in table_copy.colnames:
                raise FeatureGeneratorException(f"{name} is already a column of table.")
            try:
                table_copy[name] = result
            except Exception as err:
                raise err

        return table.__class__(table_copy)  # ensure the return type is what is expected

    def __len__(self):
        return len(self.features)
