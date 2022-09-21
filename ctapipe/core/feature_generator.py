"""
Generate Features.
"""
from .component import Component
from .expression_engine import ExpressionEngine
from .traits import List, Tuple, Unicode


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
            "You can use `numpy` as `np` and `astropy.units` as `u`. "
            "Several math functions are usable without the `np`-prefix. "
            "Use `feature.quantity.to_value(unit)` to create features without units."
        ),
    ).tag(config=True)

    def __init__(self, config=None, parent=None, **kwargs):
        super().__init__(config=config, parent=parent, **kwargs)
        self.engine = ExpressionEngine(expressions=self.features)
        self._feature_names = [name for name, _ in self.features]

    def __call__(self, table):
        for result, name in zip(self.engine(table), self._feature_names):
            if name in table.colnames:
                raise FeatureGeneratorException(f"{name} is already a column of table.")
            try:
                table[name] = result
            except Exception as err:
                raise err
        return table

    def __len__(self):
        return len(self.features)
