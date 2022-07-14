from .component import Component
from .expression_engine import ALLOWED_GLOBALS, ExpressionEngine
from .traits import List


class FeatureGeneratorException(TypeError):
    """Signal a problem with a user-defined selection criteria function"""


class FeatureGenerator(Component):
    features = List().tag(config=True)

    def __init__(self, config=None, parent=None, **kwargs):
        super().__init__(config=config, parent=parent, **kwargs)

        self.expression_engine = ExpressionEngine(
            parent=self, expressions=self.features
        )
        self.expressions = self.expression_engine()
        self.feature_names = [n for n, _ in self.features]

    def __call__(self, table):
        for e, n in zip(self.expressions, self.feature_names):
            if n in table.colnames:
                raise FeatureGeneratorException(f"{n} is already a column of table.")
            try:
                table[n] = eval(e, ALLOWED_GLOBALS, table)
            except Exception:
                raise FeatureGeneratorException(f"{e} is already a column of table.")
        return table
