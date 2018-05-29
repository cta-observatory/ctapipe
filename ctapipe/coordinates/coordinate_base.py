"""

"""
import numpy as np

__all__ = [
    'BaseCoordinate',
    'TransformationError'
]


class TransformationError(RuntimeError):
    pass


class BaseCoordinate:
    """
    Base class that all coordinate frames should inherit from, includes the code for
    finding the correct path between systems and then performing the coordinate
    transformations between them
    """

    system_order, transformations, reverse_transformations = None, None, None

    def _find_path(self, new_system):
        """

        Parameters
        ----------
        new_system: BaseCoordinate
            System to which we want to convert to

        Returns
        -------
        list: Path of function calls required to get to new system
        """
        path_start = np.where(self.system_order == self.__class__.__name__)[0][0]
        if not isinstance(new_system, str):
            new_system = new_system.__class__.__name__
        path_end = np.where(self.system_order == new_system)[0][0]
        if path_end.size == 0:
            raise TransformationError(new_system + "is not a recognised coordinate frame")

        if path_start < path_end:
            return self.transformations[path_start:path_end]
        else:
            return list(reversed(self.reverse_transformations[path_end:path_start]))

    def transform_to(self, new_system):
        """

        Parameters
        ----------
        new_system: BaseCoordinate
            System to which we want to convert to

        Returns
        -------
        BaseCoordinate: transformed coordinates
        """

        transform_path = self._find_path(new_system)
        frame = self
        for step in transform_path:
            frame = step(self)

        return frame


