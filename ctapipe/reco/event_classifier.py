import numpy as np

from astropy import units as u

from sklearn.ensemble import RandomForestClassifier

from ctapipe.reco.regressor_classifier_base import RegressorClassifierBase


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
    return 10*x**3 - 15*x**4 + 6*x**5


class EventClassifier(RegressorClassifierBase):
    def __init__(self, classifier=RandomForestClassifier,
                 cam_id_list=("cam"), **kwargs):
        super().__init__(model=classifier, cam_id_list=cam_id_list, **kwargs)

    def predict_proba_by_event(self, X):
        predict_proba = []
        for evt in X:
            tel_probas = []
            for cam_id, tels in evt.items():
                tel_probas = np.append(tel_probas,
                                       self.model_dict[cam_id].predict_proba(tels),
                                       axis=0) if tel_probas else \
                             self.model_dict[cam_id].predict_proba(tels)

            predict_proba.append(np.mean(proba_drifting(tel_probas), axis=0))

        return np.array(predict_proba)

    def predict_by_event(self, X):
        proba = self.predict_proba_by_event(X)
        predictions = self.classes_[np.argmax(proba, axis=1)]
        return predictions
