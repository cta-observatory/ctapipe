"""

"""
import numpy as np

__all__ = [
    'BaseCoordinate',
]


class BaseCoordinate:
    """

    """

    #
    system_order, transformations, reverse_transformations = None, None, None

    def init(self):
        return

    def _find_path(self, new_system):
        """

        Parameters
        ----------
        self
        new_system

        Returns
        -------

        """
        path_start = np.where(self.system_order == self.__class__.__name__)[0][0]
        path_end = np.where(self.system_order == new_system)[0][0]

        if path_start < path_end:
            return self.transformations[path_start:path_end]
        else:
            return reversed(self.reverse_transformations[path_start:path_end])

    def transform_to(self, new_system):
        """

        Parameters
        ----------
        self
        new_system

        Returns
        -------

        """

        transform_path = self._find_path(new_system)

        frame = self
        for step in transform_path:
            frame = step(self)

        return frame


