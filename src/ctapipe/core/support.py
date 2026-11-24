"""Module implementing supporting classes needed in ctapipe.core"""

__all__ = [
    "Singleton",
]


class Singleton(type):
    """metaclass for singleton pattern"""

    instance = None

    def __call__(cls, *args, **kw):
        """
        Return the single instance of this class, constructing it on first use.

        Parameters
        ----------
        *args
            Positional arguments passed to the constructor on first use.
        **kw
            Keyword arguments passed to the constructor on first use.

        Returns
        -------
        object
            The cached instance of the class.
        """
        if not cls.instance:
            cls.instance = super().__call__(*args, **kw)
        return cls.instance
