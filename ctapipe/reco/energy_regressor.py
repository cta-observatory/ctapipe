import numpy as np

from astropy import units as u

from .regressor_classifier_base import RegressorClassifierBase
from sklearn.ensemble import RandomForestRegressor


class EnergyRegressor(RegressorClassifierBase):
    """This class collects one regressor for every camera type -- given
    by `cam_id_list` -- to get an estimate for the energy of an
    air-shower.  The class interfaces with `scikit-learn` in that it
    relays all function calls not exeplicitly defined here through
    `.__getattr__` to any regressor in its collection.  This gives us
    the full power and convenience of scikit-learn and we only have to
    bother about the high-level interface.

    The fitting approach is to train on single images; then separately
    predict an image's energy and combine the different predictions
    for all the telscopes in each event in one single energy estimate.

    TODO: come up with a better merging approach to combine the
          estimators of the various camera types

    Parameters
    ----------
    regressor : scikit-learn regressor
        the regressor you want to use to estimate the shower energies
    cam_id_list : list of strings
        list of identifiers to differentiate the various sources of
        the images; could be the camera IDs or even telescope IDs. We
        will train one regressor for each of the identifiers.
    unit : astropy.Quantity
        scikit-learn regressors don't work with astropy unit. so, tell
        in advance in which unit we want to deal here.
    kwargs
        arguments to be passed on to the constructor of the regressors

    """

    def __init__(self, regressor=RandomForestRegressor,
                 cam_id_list="cam", unit=u.TeV, **kwargs):
        super().__init__(model=regressor, cam_id_list=cam_id_list,
                         unit=unit, **kwargs)

    def predict_by_event(self, event_list):
        """expects a list of events where every "event" is a dictionary
        mapping telescope identifiers to the list of feature-lists by
        the telescopes of that type in this event. Refer to the Note
        section in `.reshuffle_event_list` for an description how `event_list`
        is supposed to look like.  The singular estimate for the event
        is simply the mean of the various estimators of the event.

        event_list : list of "events"
            cf. `.reshuffle_event_list` under Notes

        Returns
        -------
        dict :
            dictionary that contains various statistical modes (mean,
            median, standard deviation) of the predicted quantity of
            every telescope for all events.

        Raises
        ------
        KeyError:
            if there is a telescope identifier in `event_list` that is not a
            key in the regressor dictionary

        """

        predict_mean = []
        predict_median = []
        predict_std = []
        for evt in event_list:
            predicts = []
            weights = []
            for cam_id, tels in evt.items():
                try:
                    t_res = self.model_dict[cam_id].predict(tels).tolist()
                    predicts += t_res
                except KeyError:
                    # QUESTION if there is no trained classifier for
                    # `cam_id`, raise an error or just pass this
                    # camera type?
                    raise KeyError("cam_id '{}' in event_list but no model defined: {}"
                                   .format(cam_id, [k for k in self.model_dict]))

                try:
                    # if a `namedtuple` is provided, we can weight the different images
                    # using some of the provided features
                    weights += [t.sum_signal_cam / t.impact_dist for t in tels]
                except AttributeError:
                    # otherwise give every image the same weight
                    weights += [1] * len(tels)

            predict_mean.append(np.average(predicts, weights=weights))
            predict_median.append(np.median(predicts))
            predict_std.append(np.std(predicts))

        return {"mean": np.array(predict_mean) * self.unit,
                "median": np.array(predict_median) * self.unit,
                "std": np.array(predict_std) * self.unit}

    def predict_by_telescope_type(self, event_list):
        """same as `predict_dict` only that it returns a list of dictionaries
        with an estimate for the target quantity for every telescope
        type separately.

        more for testing- and performance-measuring-purpouses -- to
        see how the different telescope types behave throughout the
        energy ranges and if a better (energy-dependant) combination
        of the separate telescope-wise estimators (compared to the
        mean) can be achieved.

        """

        predict_list_dict = []
        for evt in event_list:
            res_dict = {}
            for cam_id, tels in evt.items():
                t_res = self.model_dict[cam_id].predict(tels).tolist()
                res_dict[cam_id] = np.mean(t_res) * self.unit
            predict_list_dict.append(res_dict)

        return predict_list_dict

    @classmethod
    def load(cls, path, cam_id_list, unit=u.TeV):
        """this is only here to overwrite the unit argument with an astropy
        quantity

        Parameters
        ----------
        path : string
            the path where the pre-trained, pickled regressors are
            stored `path` is assumed to contain a `{cam_id}` keyword
            to be replaced by each camera identifier in `cam_id_list`
            (or at least a naked `{}`).
        cam_id_list : list
            list of camera identifiers like telescope ID or camera ID
            and the assumed distinguishing feature in the filenames of
            the various pickled regressors.
        unit : astropy.Quantity
            scikit-learn regressor do not work with units. so append
            this one to the predictions. assuming that the models
            where trained with consistent units. (default: u.TeV)

        Returns
        -------
        EnergyRegressor:
            a ready-to-use instance of this class to predict any
            quantity you have trained for

        """
        return super().load(path, cam_id_list, unit)
