from astropy.table import Table

from collections import OrderedDict


class UndefinedCut(Exception):
    pass


class PureCountingCut(Exception):
    pass


class CutFlow:
    """
    a class that keeps track of e.g. events/images that passed cuts or other
    events that could reject them """

    def __init__(self, name="CutFlow"):
        """
        Parameters
        ----------
        name : string (default: "CutFlow")
            name for the specific instance
        """
        self.cuts = OrderedDict()
        self.name = name

    def count(self, cut, weight=1):
        """
        counts an event/image at a given stage of the analysis

        Parameters
        ----------
        cut : string
            name of the cut/stage where you want to count
        weight : int or float, optional (default: 1)
            weight of the current element

        Notes
        -----
        If `cut` is not yet being tracked, it will simply be added
        Will be an alias to __getitem__
        """
        if cut not in self.cuts:
            self.cuts[cut] = [None, weight]
        else:
            self.cuts[cut][1] += weight

    def set_cut(self, cut, function):
        """
        sets a function that selects on whatever you want to count
        sets the counter corresponding to the selection criterion to 0
        that means: it overwrites whatever you counted before under this
        name

        Parameters
        ----------
        cut : string
            name of the cut/stage where you want to count
        function : function
            a function that is your selection criterion

        Notes
        -----
        add_cut and set_cut are aliases
        """
        self.cuts[cut] = [function, 0]

    def set_cuts(self, cut_dict, clear=False):
        """
        sets functions that select on whatever you want to count
        sets the counter corresponding to the selection criterion to 0
        that means: it overwrites whatever you counted before under this
        name

        Parameters
        ----------
        cut_dict : {string: functor} dictionary
            dictionary of {name: function} of cuts to add as your selection criteria
        clear : bool, optional (default: False)
            if set to `True`, clear the cut-dictionary before adding the new cuts

        Notes
        -----
        add_cuts and set_cuts are aliases
        """

        if clear:
            self.cuts = OrderedDict()

        for cut, function in cut_dict.items():
            self.cuts[cut] = [function, 0]

    def _check_cut(self, cut):
        """
        checks if `cut` is a valid name for a function to select on

        Parameters
        ----------
        cut : string
            name of the selection criterion

        Raises
        ------
        UndefinedCut if `cut` is not known
        PureCountingCut if `cut` has no associated function
        (i.e. manual counting mode)
        """

        if cut not in self.cuts:
            raise UndefinedCut(
                "unknown cut '{}' -- only know: {}"
                .format(cut, [a for a in self.cuts.keys()]))
        elif self.cuts[cut][0] is None:
            raise PureCountingCut(
                f"'{cut}' has no function associated")

    def cut(self, cut, *args, weight=1, **kwargs):
        """
        selects the function associated with `cut` and hands it all
        additional arguments provided. if the function returns `False`,
        the event counter is incremented.

        Parameters
        ----------
        cut : string
            name of the selection criterion
        args, kwargs: additional arguments
            anything you want to hand to the associated function
        weight : int or float, optional (default: 1)
            weight of the current element

        Returns
        -------
        True if the function evaluats to True
        False otherwise

        Raises
        ------
        UndefinedCut if `cut` is not known
        PureCountingCut if `cut` has no associated function
        (i.e. manual counting mode)
        """

        self._check_cut(cut)

        if self.cuts[cut][0](*args, **kwargs):
            return True
        else:
            self.cuts[cut][1] += weight
            return False

    def keep(self, cut, *args, weight=1, **kwargs):
        """
        selects the function associated with `cut` and hands it all
        additional arguments provided. if the function returns True,
        the event counter is incremented.

        Parameters
        ----------
        cut : string
            name of the selection criterion
        args, kwargs: additional arguments
            anything you want to hand to the associated function
        weight : int or float, optional (default: 1)
            weight of the current element

        Returns
        -------
        True if the function evaluats to True
        False otherwise

        Raises
        ------
        UndefinedCut if `cut` is not known
        PureCountingCut if `cut` has no associated function
        (i.e. manual counting mode)
        """

        self._check_cut(cut)

        if self.cuts[cut][0](*args, **kwargs):
            self.cuts[cut][1] += weight
            return True
        else:
            return False

    def __call__(self, *args, **kwargs):
        """
        creates an astropy table of the cut names, counted events and
        selection efficiencies
        prints the instance name and the astropy table

        Parameters
        ----------
        kwargs : keyword arguments
            arguments to be passed to the `get_table` function; see there

        Returns
        -------
        t : `astropy.table.Table`
            the table containing the cut names, counted events and
            efficiencies -- sorted in the order the cuts were added if not
            specified otherwise
        """
        print(self.name)
        t = self.get_table(*args, **kwargs)
        print(t)
        return t

    def get_table(self, base_cut=None, sort_column=None,
                  sort_reverse=False, format='5.3f'):
        """
        creates an astropy table of the cut names, counted events and
        selection efficiencies

        Parameters
        ----------
        base_cut : string, optional (default: None)
            name of the selection criterion that should be taken as 100 %
            in efficiency calculation
            if not given, the criterion with the highest count is used
        sort_column : integer, optional (default: None)
            the index of the column that should be used for sorting the entries
            by default the table is sorted in the order the cuts were added
            (index 0: cut name, index 1: number of passed events, index 2: efficiency)
        sort_reverse : bool, optional (default: False)
            if true, revert the order of the entries
        format : string, optional (default: '5.3f')
            formatting string for the efficiency column

        Returns
        -------
        t : `astropy.table.Table`
            the table containing the cut names, counted events and
            efficiencies -- sorted in the order the cuts were added if not
            specified otherwise
        """

        if base_cut is None:
            base_value = max([a[1] for a in self.cuts.values()])
        elif base_cut not in self.cuts:
            raise UndefinedCut(
                "unknown cut '{}' -- only know: {}"
                .format(base_cut, [a for a in self.cuts.keys()]))
        else:
            base_value = self.cuts[base_cut][1]

        t = Table([[cut for cut in self.cuts.keys()],
                   [self.cuts[cut][1] for cut in self.cuts.keys()],
                   [self.cuts[cut][1] / base_value for cut in self.cuts.keys()]],
                  names=['Cut Name', 'selected Events', 'Efficiency'])
        t['Efficiency'].format = format

        if sort_column is not None:
            t.sort(t.colnames[sort_column])
        # if sorted by column 0 (i.e. the cut name) default sorting (alphabetically) is
        # fine. if sorted by column 1 or 2 or `sort_reverse` is True,
        # revert the order of the table
        if (sort_column is not None and sort_column > 0) != sort_reverse:
            t.reverse()
        return t

    add_cut = set_cut
    add_cuts = set_cuts
    __getitem__ = count
