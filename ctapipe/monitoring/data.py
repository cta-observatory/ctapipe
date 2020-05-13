from sortedcontainers import SortedDict


class MonitoringData(SortedDict):

    def closest(self, lookup):
        if len(self) == 0:
            raise KeyError(f'No items in {self.__class__.__name__}')

        if len(self) == 1:
            return next(iter(self.items()))

        before, after = self.bracket(lookup)

        if before[0] is None:
            return after

        if after[0] is None:
            return before

        return min(before, after, key=lambda t: abs(t[0] - lookup))

    def before(self, lookup, inclusive=True):
        if len(self) == 0:
            raise KeyError(f'No items in {self.__class__.__name__}')

        key = next(self.irange(
            maximum=lookup, reverse=True, inclusive=(True, inclusive)
        ), None)

        if key is None:
            return None, None

        return key, self[key]

    def after(self, lookup, inclusive=True):
        if len(self) == 0:
            raise KeyError('No items in {self.__class__.__name__}')

        key = next(self.irange(
            minimum=lookup, inclusive=(inclusive, True)
        ), None)

        if key is None:
            return None, None

        return key, self[key]

    def bracket(self, lookup):
        key_before = next(self.irange(maximum=lookup, reverse=True), None)
        key_after = next(self.irange(minimum=lookup), None)

        val_before = self.get(key_before)
        val_after = self.get(key_after)

        return (key_before, val_before), (key_after, val_after)

    def interpolate_linear(self, lookup):
        if len(self) < 2:
            raise KeyError('Need at least two items for linear interpolation')

        before, after = self.bracket(lookup)

        # we found a support point exactly
        if before == after:
            return before[1]

        if before[0] is None:
            # extrapolation to earlier values
            before, after = after, self.after(after[0], inclusive=False)

        elif after[0] is None:
            before, after = self.before(before[0], inclusive=False), before

        x0, y0 = before
        x1, y1 = after
        return y0 + (lookup - x0) / (x1 - x0) * (y1 - y0)
