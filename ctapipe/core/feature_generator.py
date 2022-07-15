"""
Generate Features.
"""
from .component import Component
from .expression_engine import ExpressionEngine
from .traits import Dict


class FeatureGeneratorException(TypeError):
    """Signal a problem with a user-defined selection criteria function"""


class FeatureGenerator(Component):
    """
    Generate features for astropy.table.Table.

    Raises Exceptions in two cases:
    1. If a feature already exists in the table
    2. If a feature cannot be built with the given expression
    """

    features = Dict(
        help=(
            "Keys are the names for the new features."
            " Values are the expressions that generate the new feature."
        )
    ).tag(config=True)

    def __init__(self, config=None, parent=None, **kwargs):
        super().__init__(config=config, parent=parent, **kwargs)
        self.engine = ExpressionEngine(parent=self, expressions=self.features)

    def __call__(self, table):
        for result, name in zip(self.engine(table), self.features.keys()):
            if name in table.colnames:
                raise FeatureGeneratorException(f"{name} is already a column of table.")
            try:
                table[name] = result
            except Exception as err:
                raise err
        return table
