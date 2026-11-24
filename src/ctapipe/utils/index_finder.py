"""Find the index of the nearest value in a possibly unsorted sequence."""

import bisect
import warnings

import numpy as np

__all__ = ["IndexFinder"]


class IndexFinder:
    """
    Find the index of the closest value in a possibly unsorted sequence.

    The lookup uses Python's bisect module on sorted values and returns indices into the
    original sequence. Duplicate values trigger a warning and retain their last index.
    The helper is used, for example, to match a pointing to an event's trigger time.


    Explanations can be found here:
    https://stackoverflow.com/questions/9706041/finding-index-of-an-item-closest-to-the-value-in-a-list-thats-not-entirely-sort

    Since we only need the searching part, the code has been cut down slightly.

    Parameters
    ----------
    values : sequence of float
        Numeric values to index. Indices refer to the original input order.
    """

    def __init__(self, values):
        if len(np.unique(values)) != len(values):
            warnings.warn("Duplicated values in IndexFinder")

        self.numindexes = dict((val, n) for n, val in enumerate(values))
        self.nums = sorted(self.numindexes)

    def _rank(self, target):
        """
        Find the closest value's position in the sorted list.

        On a tie, prefer the larger value. Use `closest` to get its index in the
        original sequence.
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
        Return the index of the closest value in the original sequence.

        Parameters
        ----------
        target : float
            Value for which to find the nearest entry in the indexed sequence.
        """
        if len(self.nums) == 1:
            return 0
        try:
            return self.numindexes[self.nums[self._rank(target)]]
        except IndexError:
            return 0
