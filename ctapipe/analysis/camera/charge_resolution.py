import numpy as np
import pandas as pd

__all__ = ['ChargeResolutionCalculator']


class ChargeResolutionCalculator:
    """
    Calculates the charge resolution with an efficient, low-memory,
    interative approach, allowing the contribution of data/events
    without reading the entire dataset into memory.

    Utilises Pandas DataFrames, and makes no assumptions on the order of
    the data, and does not require the true charge to be integer (as may
    be the case for lab measurements where an average illumination
    is used).

    A list is filled with a dataframe for each contribution, and only
    amalgamated into a single dataframe (reducing memory) once the memory
    of the list becomes large (or at the end of the filling),
    reducing the time required to produce the output.

    Parameters
    ----------
    mc_true : bool
        Indicate if the "true charge" values are from the sim_telarray
        files, and therefore without poisson error. The poisson error will
        therefore be included in the charge resolution calculation.

    Attributes
    ----------
    self._mc_true : bool
    self._df_list : list
    self._df : pd.DataFrame
    self._n_bytes : int
        Monitors the number of bytes being held in memory
    """

    def __init__(self, mc_true=True):
        self._mc_true = mc_true
        self._df_list = []
        self._df = pd.DataFrame()
        self._n_bytes = 0
        self._max_bytes = 1E9

    @staticmethod
    def rmse_abs(sum_, n):
        return np.sqrt(sum_ / n)

    @staticmethod
    def rmse(true, sum_, n):
        return ChargeResolutionCalculator.rmse_abs(sum_, n) / np.abs(true)

    @staticmethod
    def charge_res_abs(true, sum_, n):
        return np.sqrt((sum_ / n) + true)

    @staticmethod
    def charge_res(true, sum_, n):
        return (ChargeResolutionCalculator.charge_res_abs(true, sum_, n)
                / np.abs(true))

    def add(self, pixel, true, measured):
        """
        Contribute additional values to the Charge Resolution

        Parameters
        ----------
        pixel : ndarray
            1D array containing the pixel for each entry
        true : ndarray
            1D array containing the true charge for each entry
        measured : ndarray
            1D array containing the measured charge for each entry
        """
        diff2 = np.power(measured - true, 2)
        df = pd.DataFrame(dict(
            pixel=pixel,
            true=true,
            sum=diff2,
            n=np.uint32(1)
        ))
        self._df_list.append(df)
        self._n_bytes += df.memory_usage(index=True, deep=True).sum()
        if self._n_bytes > self._max_bytes:
            self._amalgamate()

    def _amalgamate(self):
        """
        Concatenate the dataframes inside the list, and sum together
        values per pixel and true charge in order to reduce memory use.
        """
        self._df = pd.concat([self._df, *self._df_list], ignore_index=True)
        self._df = self._df.groupby(['pixel', 'true']).sum().reset_index()
        self._n_bytes = 0
        self._df_list = []

    def finish(self):
        """
        Perform the final amalgamation, and calculate the charge resolution
        from the resulting sums

        Returns
        -------
        df_p : pd.DataFrame
            Dataframe containing the charge resolution per pixel
        df_c : pd.DataFrame
            Dataframe containing the charge resolution for the entire camera
        """
        self._amalgamate()

        self._df = self._df.loc[self._df['true'] != 0]

        df_p = self._df.copy()
        true = df_p['true'].values
        sum_ = df_p['sum'].values
        n = df_p['n'].values
        if self._mc_true:
            df_p['charge_resolution'] = self.charge_res(true, sum_, n)
            df_p['charge_resolution_abs'] = self.charge_res_abs(true, sum_, n)
        else:
            df_p['charge_resolution'] = self.rmse(true, sum_, n)
            df_p['charge_resolution_abs'] = self.rmse_abs(sum_, n)
        df_c = self._df.copy().groupby('true').sum().reset_index()
        df_c = df_c.drop(columns='pixel')
        true = df_c['true'].values
        sum_ = df_c['sum'].values
        n = df_c['n'].values
        if self._mc_true:
            df_c['charge_resolution'] = self.charge_res(true, sum_, n)
            df_c['charge_resolution_abs'] = self.charge_res_abs(true, sum_, n)
        else:
            df_c['charge_resolution'] = self.rmse(true, sum_, n)
            df_c['charge_resolution_abs'] = self.rmse_abs(sum_, n)

        return df_p, df_c
