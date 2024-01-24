import bisect
import warnings

import numpy as np

__all__ = ["IndexFinder"]


class IndexFinder:
    """
    Helper class to find the index of the closest matching value in an array/list/...,
    used to locate the pointing of an event based on the trigger time.
    This searches using pythons bisect module.
    Duplicated values will result in the first value being returned.


    Explanations can be found here:
    https://stackoverflow.com/questions/9706041/finding-index-of-an-item-closest-to-the-value-in-a-list-thats-not-entirely-sort

    Since we only need the searching part, the code has been cut down slightly.
    """

    def __init__(self, values):
        if len(np.unique(values)) != len(values):
            warnings.warn("Duplicated values in IndexFinder")

        self.numindexes = dict((val, n) for n, val in enumerate(values))
        self.nums = sorted(self.numindexes)

    def _rank(self, target):
        """
        Searches for the closest match in the ordered
        self.nums list and returns the index relative to this
        list.
        To find the index relative to the original values, use the
        `closest` method.
        """
        rank = bisect.bisect(self.nums, target)
        if rank == 0:
            pass
        elif len(self.nums) == rank:
            rank -= 1
        else:
            dist1 = target - self.nums[rank - 1]
            dist2 = self.nums[rank] - target
            if dist1 < dist2:
                rank -= 1
        return rank

    def closest(self, target):
        """
        Given a value, that is comparable to the
        associated ``values`` list, return the
        index of the closest matching entry relative to the unordered
        list, that was given at construction.
        """
        if len(self.nums) == 1:
            return 0
        try:
            return self.numindexes[self.nums[self._rank(target)]]
        except IndexError:
            return 0
