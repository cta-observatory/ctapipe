import numpy as np
from math import factorial
from scipy.special import binom
from numba import jit
from ctapipe.analysis.camera.spe_fitter import SPEFitter


class PMTFitter(SPEFitter):
    def __init__(self, n_illuminations, config_path=None):
        """
        SpectrumFitter which uses the PMT fitting formula

        Parameters
        ----------
        n_illuminations : int
            Number of illuminations to fit simultaneously
        """
        super().__init__(n_illuminations, config_path)

        self.nbins = 100
        self.range = [-40, 150]

        self.add_parameter("norm", None, 0, 100000, fix=True, multi=True)
        self.add_parameter("eped", 0, -10, 10)
        self.add_parameter("eped_sigma", 9, 2, 20)
        self.add_parameter("spe", 25, 5, 30)
        self.add_parameter("spe_sigma", 2, 1, 20)
        self.add_parameter("lambda_", 0.2, 0.001, 6, multi=True)

    def _prepare_params(self, p0, limits, fix):
        for i in range(self.n_illuminations):
            norm = 'norm{}'.format(i)
            if p0[norm] is None:
                p0[norm] = np.trapz(self.hist[i], self.between)

    @staticmethod
    def _fit(x, **kwargs):
        return pmt_spe_fit(x, **kwargs)


C = np.sqrt(2.0 * np.pi)
FACTORIAL = np.array([factorial(i) for i in range(15)])
FACTORIAL_0 = FACTORIAL[0]
NPEAKS = 11
N = np.arange(NPEAKS)[:, None]
J = np.arange(NPEAKS)[None, :]
K = np.arange(1, NPEAKS)[:, None]
FACTORIAL_J_INV = 1 / FACTORIAL[J]
BINOM = binom(N - 1, J - 1)


@jit(nopython=True)
def _normal_pdf(x, mean=0, std_deviation=1):
    """
    Evaluate the normal probability density function. Faster than
    `scipy.norm.pdf` as it does not check the inputs.

    Parameters
    ----------
    x : ndarray
        The normal probability function will be evaluated
    mean : float or ndarray
    std_deviation : float or array_like

    Returns
    -------
    ndarray
    """
    u = (x - mean) / std_deviation
    return np.exp(-0.5 * u ** 2) / (C * std_deviation)


@jit(nopython=True)
def _poisson_pmf_j(mu):
    """
    Evaluate the poisson pmf for a fixed k=J events

    Parameters
    ----------
    mu : Average number of events

    Returns
    -------
    ndarray
    """
    return mu ** J * np.exp(-mu) * FACTORIAL_J_INV


@jit(nopython=True)
def pedestal_signal(x, norm, eped, eped_sigma, lambda_):
    """
    Obtain the signal provided by the pedestal in the pulse spectrum.

    Parameters
    ----------
    x : ndarray
        The x values to evaluate at
    norm : float
        Integral of the zeroth peak in the distribution, represents p(0)
    eped : float
        Distance of the zeroth peak from the origin
    eped_sigma : float
        Sigma of the zeroth peak, represents electronic noise of the system
    lambda_ : float
        Poisson mean

    Returns
    -------
    signal : ndarray
        The y values of the signal provided by the pedestal.
    """
    p_ped = np.exp(-lambda_)  # Poisson PMF for k = 0, mu = lambda_
    signal = norm * p_ped * _normal_pdf(x, eped, eped_sigma)
    return signal


# @jit
def pe_signal(k, x, norm, eped, eped_sigma, spe, spe_sigma, lambda_):
    """
    Obtain the signal provided by photoelectrons in the pulse spectrum.

    Parameters
    ----------
    k : int or ndarray
        The NPEs to evaluate. A list of NPEs can be passed here, provided it
        is broadcast as [:, None], and the x input is broadcast as [None, :],
        the return value will then be a shape [k.size, x.size].
        k must be greater than or equal to 1.
    x : ndarray
        The x values to evaluate at
    norm : float
        Integral of the zeroth peak in the distribution, represents p(0)
    eped : float
        Distance of the zeroth peak from the origin
    eped_sigma : float
        Sigma of the zeroth peak, represents electronic noise of the system
    spe : float
        Signal produced by 1 photo-electron
    spe_sigma : float
        Spread in the number of photo-electrons incident on the MAPMT
    lambda_ : float
        Poisson mean (illumination in p.e.)

    Returns
    -------
    signal : ndarray
        The y values of the signal provided by the photoelectrons. If k is an
        integer, this will have same shape as x. If k is an array,
        and k and x are broadcase correctly, this will have
        shape [k.size, x.size].

    """
    # Obtain poisson distribution
    p = _poisson_pmf_j(lambda_)[0]

    pe_sigma = np.sqrt(k * spe_sigma ** 2 + eped_sigma ** 2)

    signal = norm * p[k] * _normal_pdf(x, eped + k * spe, pe_sigma)

    return signal


def pmt_spe_fit(x, norm, eped, eped_sigma, spe, spe_sigma, lambda_, **kwargs):
    """
    Fit for the SPE spectrum of a MAPM

    Parameters
    ----------
    x : 1darray
        The x values to evaluate at
    norm : float
        Integral of the zeroth peak in the distribution, represents p(0)
    eped : float
        Distance of the zeroth peak from the origin
    eped_sigma : float
        Sigma of the zeroth peak, represents electronic noise of the system
    spe : float
        Signal produced by 1 photo-electron
    spe_sigma : float
        Spread in the number of photo-electrons incident on the MAPMT
    lambda_ : float
        Poisson mean (illumination in p.e.)

    Returns
    -------
    signal : ndarray
        The y values of the total signal.
    """

    # Obtain pedestal signal
    params = [norm, eped, eped_sigma, lambda_]
    ped_s = pedestal_signal(x, *params)

    # Obtain pe signal

    params = [norm, eped, eped_sigma, spe, spe_sigma, lambda_]
    pe_s = pe_signal(K, x[None, :], *params).sum(0)

    signal = ped_s + pe_s

    return signal
