import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .regressor_classifier_base import RegressorClassifierBase

__all__ = ["proba_drifting", "EventClassifier"]


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
    def __init__(
        self, classifier=RandomForestClassifier, cam_id_list=("cam"), **kwargs
    ):
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
                    tel_weights += [t.sum_signal_cam / t.impact_dist for t in tels]
                except AttributeError:
                    # otherwise give every image the same weight
                    tel_weights += [1] * len(tels)

            predict_proba.append(
                np.average(proba_drifting(tel_probas), weights=tel_weights, axis=0)
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
