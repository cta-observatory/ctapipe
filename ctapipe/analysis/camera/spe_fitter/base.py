from abc import abstractmethod
import iminuit
import numpy as np
from scipy.stats.distributions import poisson
import yaml
import warnings


class SPEFitterMeta(type):
    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj._post_init()
        return obj


class SPEFitter(metaclass=SPEFitterMeta):
    def __init__(self, n_illuminations, config_path=None):
        """
        Base class for fitters of Single-Photoelectron spectra. Built to
        flexibly handle any number of illuminations simultaneously.

        Parameters
        ----------
        n_illuminations : int
            Number of illuminations to fit simultaneously
        config_path : str
            Path to JAML config file
        """
        self.hist = None
        self.edges = None
        self.between = None
        self.coeff = None
        self.errors = None
        self.p0 = None

        self.nbins = 100
        self.range = [-10, 100]

        self.coeff_names = []
        self.multi_coeff = []
        self.initial = dict()
        self.limits = dict()
        self.fix = dict()

        self.n_illuminations = n_illuminations
        self.config_path = config_path

    def _post_init(self):
        if self.config_path:
            self.load_config(self.config_path)

    @property
    def fit_x(self):
        """
        Default X coordinates for the fit

        Returns
        -------
        ndarray
        """
        return np.linspace(self.edges[0], self.edges[-1], 10*self.edges.size)

    @property
    def fit(self):
        """
        Curve for the current fit result

        Returns
        -------
        ndarray
        """
        return self.fit_function(x=self.fit_x, **self.coeff)


    def add_parameter(self, name, initial, lower, upper,
                      fix=False, multi=False):
        """
        Add a new parameter for this particular fit function

        Parameters
        ----------
        name : str
            Name of the parameter
        initial : float
            Initial value for the parameter
        lower : float
            Lower limit for the parameter
        upper : float
            Upper limit for the parameter
        fix : bool
            Specify if the parameter should be fixed
        multi : bool
            Specify if the parameter should be duplicated for additional
            illuminations
        """
        if not multi:
            self.coeff_names.append(name)
            self.initial[name] = initial
            self.limits["limit_" + name] = (lower, upper)
            self.fix["fix_" + name] = fix
        else:
            self.multi_coeff.append(name)
            for i in range(self.n_illuminations):
                name_i = name + str(i)
                self.coeff_names.append(name_i)
                self.initial[name_i] = initial
                self.limits["limit_" + name_i] = (lower, upper)
                self.fix["fix_" + name_i] = fix
        # ds = "minimize_function(" + ", ".join(self.coeff_names) + ")"
        # self._minimize_function.__func__.__doc__ = ds

    def load_config(self, path):
        """
        Load a YAML configuration file to set initial fitting parameters

        Parameters
        ----------
        path : str
        """
        print("Loading SpectrumFitter configuration from: {}".format(path))
        with open(path, 'r') as file:
            d = yaml.safe_load(file)
            if d is None:
                return
            self.nbins = d.pop('nbins', self.nbins)
            self.range = d.pop('range', self.range)
            for c in self.coeff_names:
                if 'initial' in d:
                    ini = c
                    self.initial[ini] = d['initial'].pop(c, self.initial[ini])
                    if(self.initial[ini] is not None):
                        self.initial[ini] = float(self.initial[ini])
                if 'limits' in d:
                    lim = "limit_" + c
                    list_ = d['limits'].pop(c, self.limits[lim])
                    if(isinstance(list_,list)):
                        self.limits[lim] = tuple([float(l) for l in list_])
                    else:
                        self.limits[lim] = list_
                if 'fix' in d:
                    fix = "fix_" + c
                    self.fix[fix] = d['fix'].pop(c, self.fix[fix])
            if 'initial' in d and not d['initial']:
                d.pop('initial')
            if 'limits' in d and not d['limits']:
                d.pop('limits')
            if 'fix' in d and not d['fix']:
                d.pop('fix')
            if d:
                print("WARNING: Unused SpectrumFitter config parameters:")
                print(d)

    def save_config(self, path):
        """
        Save the configuration of the fit. If the fit has already been
        performed, the fit coefficients will be included as the initial
        coefficients

        Parameters
        ----------
        path : str
            Path to save the configuration file to
        """
        print("Writing SpectrumFitter configuration to: {}".format(path))
        initial = dict()
        limits = dict()
        fix = dict()
        coeff_dict = self.coeff if self.coeff else self.initial
        for c, val in coeff_dict.items():
            initial[c] = val
        for c, val in self.limits.items():
            limits[c.replace("limit_", "")] = val
        for c, val in self.fix.items():
            fix[c.replace("fix_", "")] = val
        data = dict(
            nbins=self.nbins,
            range=self.range,
            initial=initial,
            limits=limits,
            fix=fix
        )
        with open(path, 'w') as outfile:
            yaml.safe_dump(data, outfile, default_flow_style=False)

    def apply(self, *charges):
        """
        Fit the spectra

        Parameters
        ----------
        charges : list[ndarray]
            A list of the charges to fit. Should have a length equal to the
            self.n_illuminations.
        """
        assert len(charges) == self.n_illuminations
        bins = self.nbins
        range_ = self.range
        self.hist = []
        for i in range(self.n_illuminations):
            h, e, b = self.get_histogram(charges[i], bins, range_)
            self.hist.append(h)
            self.edges = e
            self.between = b

        self._perform_fit()

    @staticmethod
    def get_histogram(charge, bins, range_):
        """
        Obtain a histogram for the spectrum.

        Look at `np.histogram` documentation for further info on Parameters.

        Parameters
        ----------
        charge : ndarray
        bins
        range_

        Returns
        -------
        hist : ndarray
            The histogram
        edges : ndarray
            Edges of the histogram
        between : ndarray
            X values of the middle of each bin
        """
        hist, edges = np.histogram(charge, bins=bins, range=range_)
        between = (edges[1:] + edges[:-1]) / 2

        return hist, edges, between

    def _perform_fit(self):
        """
        Run iminuit on the fit function to find the best fit
        """
        self.coeff = {}
        self.p0 = self.initial.copy()
        limits = self.limits.copy()
        fix = self.fix.copy()
        self._prepare_params(self.p0, limits, fix)

        m0 = iminuit.Minuit(self._minimize_function,
                            **self.p0, **limits, **fix,
                            print_level=0, pedantic=False, throw_nan=True,
                            forced_parameters=self.coeff_names)
        m0.migrad()
        self.coeff = m0.values

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', iminuit.HesseFailedWarning)
            m0.hesse()
        self.errors = m0.errors

    def _prepare_params(self, p0, limits, fix):
        """
        Apply some automation to the contents of initial, limits, and fix
        dictionaries.

        Parameters
        ----------
        p0 : dict
            Initial values dict
        limits : dict
            Dict containing the limits for each parameter
        fix : dict
            Dict containing which parameters should be fixed
        """
        pass

    def _minimize_function(self, *args):
        """
        Function which calculates the likelihood to be minimised.

        Parameters
        ----------
        args
            The values to apply to the fit function.

        Returns
        -------
        likelihood : float
        """
        kwargs = dict(zip(self.coeff_names, args))
        x = self.between
        y = self.hist
        p = self.fit_function(x, **kwargs)
        like = [-2 * poisson._logpmf(y[i], p[i])
                for i in range(self.n_illuminations)]
        like = np.hstack(like)
        return np.nansum(like)

    def fit_function(self, x, **kwargs):
        """
        Function which applies the parameters for each illumination and
        returns the resulting curves.

        Parameters
        ----------
        x : ndarray
            X values
        kwargs
            The values to apply to the fit function

        Returns
        -------

        """
        p = []
        for i in range(self.n_illuminations):
            for coeff in self.multi_coeff:
                kwargs[coeff] = kwargs[coeff + str(i)]
            p.append(self._fit(x, **kwargs))
        return p

    @staticmethod
    @abstractmethod
    def _fit(x, **kwargs):
        """
        Define the low-level function to be used in the fit

        Parameters
        ----------
        x : ndarray
            X values
        kwargs
            The values to apply to the fit function

        Returns
        -------
        ndarray
        """
        pass
