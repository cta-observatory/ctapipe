class CTAPipeException(Exception):
    pass


class TooFewEvents(CTAPipeException):
    """Raised if something that needs a minimum number of event gets fewer"""
