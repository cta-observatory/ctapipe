class CTAPipeException(Exception):
    pass


class TooFewEvents(CTAPipeException):
    """Raised if something that needs a minimum number of event gets fewer"""


class OptionalDependencyMissing(ModuleNotFoundError):
    """Raised if an optional dependency required for a feature is not installed"""

    def __init__(self, module):
        self.module = module
        msg = f"'{module}' is required for this functionality of ctapipe"
        super().__init__(msg)
