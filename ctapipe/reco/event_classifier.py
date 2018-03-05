import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .regressor_classifier_base import RegressorClassifierBase

__all__ = ['proba_drifting','EventClassifier']


def proba_drifting(x):
    """
    gives more weight to outliers -- i.e. close to 0 and 1
    the polynomial was constructed with the following constraints:
    • f(0) = 0
    • f(0.5) = 0.5
    • f(1) = 1
    • f'(0) = 0
    • f'(0.5) = 1
    • f'(1) = 0
    """
    return 10 * x ** 3 - 15 * x ** 4 + 6 * x ** 5


class EventClassifier(RegressorClassifierBase):

    def __init__(self, classifier=RandomForestClassifier,
                 cam_id_list=("cam"), **kwargs):
        super().__init__(model=classifier, cam_id_list=cam_id_list, **kwargs)

    def predict_proba_by_event(self, X):
        predict_proba = []
        for evt in X:
            tel_probas = None
            tel_weights = []
            for cam_id, tels in evt.items():
                these_probas = self.model_dict[cam_id].predict_proba(tels)
                if tel_probas is not None:
                    tel_probas = np.append(these_probas, tel_probas, axis=0)
                else:
                    tel_probas = these_probas
                try:
                    # if a `namedtuple` is provided, we can weight the
                    # different images using some of the provided features
                    tel_weights += [t.sum_signal_cam / t.impact_dist for t in
                                    tels]
                except AttributeError:
                    # otherwise give every image the same weight
                    tel_weights += [1] * len(tels)

            predict_proba.append(
                np.average(proba_drifting(tel_probas),
                           weights=tel_weights,
                           axis=0)
            )

        return np.array(predict_proba)

    def predict_by_event(self, X):
        proba = self.predict_proba_by_event(X)
        predictions = self.classes_[np.argmax(proba, axis=1)]
        return predictions

    def compute_Qfactor(self, proba, labels: int, nbins):
        """
        Compute Q-factor for each gammaness (bin edges are 0 - 1)

        Parameters
        ----------

        proba: predicted probabilities to be a gamma!
        labels: true labels
        nbins: number of bins for gammaness

        Returns
        -------

        Q-factor array
        """
        bins = np.linspace(0, 1, nbins)

        # assuming labels are 0 for protons, 1 for gammas
        # np.nonzero function return indexes
        gammas_idx = np.nonzero(proba * labels)
        gammas = proba[gammas_idx]

        hadrons_idx = np.nonzero(proba * np.logical_not(labels))
        hadrons = proba[hadrons_idx]

        # tot number of gammas
        Ng = len(gammas)
        # of protons
        Nh = len(hadrons)

        # binning and cumsum for gammas
        gbins = pd.cut(gammas, bins)
        gcount = gbins.value_counts()
        # reverse array
        gcount[:] = gcount.values[::-1].copy()
        g_cumsum = np.cumsum(gcount.values)
        eps_g = g_cumsum / Ng

        # binning and cumsum for protons
        hbins = pd.cut(hadrons, bins)
        hcount = hbins.value_counts()
        # reverse array
        hcount[:] = hcount.values[::-1].copy()
        h_cumsum = np.cumsum(hcount.values)
        eps_h = h_cumsum / Nh

        Q = eps_g / np.sqrt(eps_h)
        Q[:] = Q[::-1].copy()

        return Q, bins[1:]  # , eps_g[::-1], eps_h[::-1]

    def _hyperBinning(self, x, featsToGroupBy: list):
        """
        This function is for hyper binning with pandas. It is intended to be
        used here in order to level number of events before training the
        classifier; for more general purposes, it is the Histogram in
        utils/fitshistogram.py to be used.

        This outputs the input array grouped in as many bins as present in
        the featsToGroupBy list. This is a list of dictionaries,
        each dictionary is related to a feature (array column) to be binned.

        Beware of bins generation: since np.linspace is used, if you want log
        bins you have to make log of the input array column and pass log(max
        and min)!

        Parameters
        ----------
        x: input array to be binned

        featsToGroupBy: list(dictionary)
              list of dictionaries (see Examples for dict keys)

        Output
        ----------
        A pandas.core.groupby.DataFrameGroupBy object


        Examples
        --------

        .. code-block: python

            featsToGroupBy = [{'feat': 'feat0',
                               'maxf': max(x[:, 0]),
                               'minf': min(x[:, 0]),
                               'col': 0,
                               'nbins': 10},
                             ...
                              {'feat': 'feat1',
                               'maxf': max(x[:, 1]),
                               'minf': min(x[:, 1]),
                               'col': 1, 'nbins': 5}]


        where

        'feat' is feature name (optional);
        'maxf' and 'minf' are the range where to bin in
        'col' is feature column number in the input array
        'nbins' is the number of requested bins
        """
        dfx = pd.DataFrame(x)
        binning_list = []

        for i in range(len(featsToGroupBy)):
            feat_dict = featsToGroupBy[i]
            bins = np.linspace(feat_dict['minf'], feat_dict['maxf'],
                               feat_dict['nbins'] + 1)
            binning_list.append(pd.cut(dfx[feat_dict['col']], bins))

        groups = dfx.groupby(binning_list)

        return groups

    def level_populations(self, group_signal, group_bgd, signal_evts, bgd_evts):
        """Equalize number of entries in each bin. When doing signal -
        background separation, it is common to wrangle input data equalizing
        the number of entries for gammas and hadron in all the requested
        hyper-bins, before training the classifier. This is different from
        the sklearn train_test_split in the sense that this level the two
        populations based upon a previous binning over any feature requested.

        Parameters
        ----------
        group_signal: pandas.core.groupby.DataFrameGroupBy
            (multi dimension) histogram of signal population in pandas groups
            format
        group_bgd: pandas.core.groupby.DataFrameGroupBy
            (multi dimension) histogram of background population in pandas
             groups format
        signal_evts: np.ndarray
            array of signal events, to be equalized
        bgd_evts: np.ndarray
            array of background events, to be equalized

        Returns
        -------
        np.ndarray:
            array of gammas and array of hadrons now of the same size

        """
        logger = logging.getLogger('level_population')
        logging.basicConfig(level=logging.DEBUG)

        try:
            type(group_signal) is pd.core.groupby.DataFrameGroupBy
        except TypeError:
            raise TypeError("This function wants pandas DataFrameGroupBy "
                            "objects as group_* inputs")
        try:
            type(group_bgd) is pd.core.groupby.DataFrameGroupBy
        except TypeError:
            raise TypeError("This function wants pandas DataFrameGroupBy "
                            "objects as group_* inputs")

        # convert to pandas to use .drop()` dataframe names follow the
        # original use case gamma (dfg) vs hadrons (dfh)
        dfg = pd.DataFrame(signal_evts)
        dfh = pd.DataFrame(bgd_evts)

        # have a unique set of keys in the histograms
        s = set(group_signal.indices)
        s.update(group_bgd.indices)

        for key in s:
            if key in group_signal.indices and key in group_bgd.indices:
                # count exceeding records
                exceeding = len(group_bgd.indices[key]) - len(
                    group_signal.indices[key])

                # drop records from dataset picking exceeding number of indices
                # among those in group.indices[key]
                if exceeding > 0:
                    logger.debug('bin: %s - g: %s\th: %s\t - Removing %s '
                                 'protons',
                                 key, len(group_signal.indices[key]),
                                 len(group_bgd.indices[key]),
                                 np.abs(exceeding))

                    r_ind_list = list(
                        np.random.choice(group_bgd.indices[key],
                                         size=exceeding,
                                         replace=False)
                    )
                    dfh.drop(r_ind_list, inplace=True)
                elif exceeding < 0:
                    logger.debug('bin: %s - g: %s\th: %s\t - Removing %s '
                                 'gammas',
                                 key,
                                 len(group_signal.indices[key]),
                                 len(group_bgd.indices[key]),
                                 np.abs(exceeding))
                    r_ind_list = list(
                        np.random.choice(group_signal.indices[key],
                                         size=-exceeding,
                                         replace=False)
                    )
                    dfg.drop(r_ind_list, inplace=True)

            elif key in group_bgd.indices and key not in group_signal.indices:
                dfh.drop(group_bgd.indices[key], inplace=True)

            elif key in group_signal.indices and key not in group_bgd.indices:
                dfg.drop(group_signal.indices[key], inplace=True)

        return np.array(dfg), np.array(dfh)
