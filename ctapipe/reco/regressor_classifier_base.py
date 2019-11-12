from copy import deepcopy

import numpy as np

from astropy import units as u
from sklearn.preprocessing import StandardScaler


class RegressorClassifierBase:
    """This class collects one model for every camera type -- given by
    `cam_id_list` -- to get an estimate for the energy of an
    air-shower.  The class interfaces with `scikit-learn` in that it
    relays all function calls not exeplicitly defined here through
    `.__getattr__` to any model in its collection.  This gives us the
    full power and convenience of scikit-learn and we only have to
    bother about the high-level interface.

    The fitting approach is to train on single images; then separately
    predict an image's energy and combine the different predictions
    for all the telscopes in each event in one single energy estimate.

    TODO: come up with a better merging approach to combine the
          estimators of the various camera types

    Parameters
    ----------

    model: scikit-learn model
        the model you want to use to estimate the shower energies
    cam_id_list: list of strings
        list of identifiers to differentiate the various sources of
        the images; could be the camera IDs or even telescope IDs. We
        will train one model for each of the identifiers.
    unit: 1 or astropy unit
        scikit-learn regressors don't work with astropy unit. so, tell
        in advance in which unit we want to deal here in case we need
        one. (default: 1)
    kwargs: **dict
        arguments to be passed on to the constructor of the regressors

    """

    def __init__(self, model, cam_id_list, unit=1, **kwargs):
        self.model_dict = {}
        self.input_features_dict = {}
        self.output_features_dict = {}
        self.unit = unit
        for cam_id in cam_id_list or []:
            self.model_dict[cam_id] = model(**deepcopy(kwargs))

    def __getattr__(self, attr):
        """We interface this class with the "first" model in `.model_dict`
        and relay all function calls over to it. This gives access to
        all the fancy implementations in the scikit-learn classes
        right from this class and we only have to overwrite the
        high-level interface functions we want to adapt to our
        use-case.

        """
        return getattr(next(iter(self.model_dict.values())), attr)

    def __str__(self):
        """ return the class name of the used model """
        return str(next(iter(self.model_dict.values()))).split("(")[0]

    def reshuffle_event_list(self, X, y):
        """I collect the data event-wise as dictionaries of the different
        camera identifiers but for the training it is more convenient
        to have a dictionary containing flat arrays of the
        feature lists. This function flattens the `X` and `y` arrays
        accordingly.

        This is only for convenience during the training and testing
        phase, later on, we won't collect data first event- and then
        telescope-wise. That's why this is a separate function and not
        implemented in `.fit`.

        Parameters
        ----------
        X : list of dictionaries of lists
            collection of training features see Notes section for a sketch of
            the container's expected shape
        y : list of astropy quantities
            list of the training targets (i.e. shower energies) for all events

        Returns
        -------
        trainFeatures, trainTarget : dictionaries of lists
            flattened containers to be handed over to `.fit`

        Raises
        ------
        KeyError:
            in case `X` contains keys that were not provided with
            `cam_id_list` during `.__init__` or `.load`.

        Notes
        -----
        `X` and `y` are expected to be lists of events:
        `X = [event_1, event_2, ..., event_n]`
        `y = [energy_1, energy_2, ..., energy_n]`

        with `event_i` being a dictionary:
        `event_i = {tel_type_1: list_of_tels_1, ...}`
        and `energy_i` is the energy of `event_i`

        `list_of_tels_i` being the list of all telescope of type
        `tel_type_1` containing the features to fit and predict the
        shower energy: `list_of_tels_i = [[tel_1_feature_1,
        tel_1_feature_2, ...], [tel_2_feature_1, tel_2_feature_2,
        ...], ...]`

        after reshuffling, the resulting lists will look like this::

          trainFeatures = {tel_type_1: [features_event_1_tel_1,
                                       features_event_1_tel_2,
                                       features_event_2_tel_1, ...],
                          tel_type_2: [features_event_1_tel_3,
                                       features_event_2_tel_3,
                                       features_event_2_tel_4, ...],
                          ...}


        `trainTarget` will be a dictionary with the same keys and a
        lists of energies corresponding to the features in the
        `trainFeatures` lists.

        """

        trainFeatures = {a: [] for a in self.model_dict}
        trainTarget = {a: [] for a in self.model_dict}

        for evt, target in zip(X, y):
            for cam_id, tels in evt.items():
                try:
                    # append the features-lists of the current event
                    # and telescope type to the flat list of
                    # features-lists for this telescope type
                    trainFeatures[cam_id] += tels
                except KeyError:
                    raise KeyError("cam_id '{}' in X but no model defined: {}"
                                   .format(cam_id, [k for k in self.model_dict]))

                try:
                    # add a target-entry for every feature-list
                    trainTarget[cam_id] += \
                        [target.to(self.unit).value] * len(tels)
                except AttributeError:
                    # in case the target is not given as an astropy
                    # quantity let's hope that the user keeps proper
                    # track of the unit themself (might be just `1` anyway)
                    trainTarget[cam_id] += [target] * len(tels)
        return trainFeatures, trainTarget

    def fit(self, X, y, sample_weight=None):
        """This function fits a model against the collected features;
        separately for every telescope identifier.

        Parameters
        ----------
        X : dictionary of lists of lists
            Dictionary that maps the telescope identifiers to lists of
            feature-lists.  The values of the dictionary are the lists
            `scikit-learn` regressors train on and are supposed to
            comply to their format requirements e.g. each featuer-list
            has to contain the same features at the same position
        y : dictionary of lists
            the energies corresponding to all the feature-lists of `X`
        sample_weight : dictionary of lists, optional (default: None)
            lists of weights for the various telescope identifiers
            Note: Not all models support this; e.g. RandomForests do
            (Regressor and Classifier)

        Returns
        -------
        self

        Raises
        ------
        KeyError:
            in case `X` contains keys that are either not in `y` or
            were not provided before with `cam_id_list`.

        """

        sample_weight = sample_weight or {}

        for cam_id in X:
            if cam_id not in y:
                raise KeyError("cam_id '{}' in X but not in y: {}"
                               .format(cam_id, [k for k in y]))

            if cam_id not in self.model_dict:
                raise KeyError("cam_id '{}' in X but no model defined: {}"
                               .format(cam_id, [k for k in self.model_dict]))

            # add a `None` entry in the weights dictionary in case there is no entry yet
            if cam_id not in sample_weight:
                sample_weight[cam_id] = None

            # for every `cam_id` train one model (as long as there are events in `X`)
            if len(X[cam_id]):
                try:
                    self.model_dict[cam_id].fit(X[cam_id], y[cam_id],
                                                sample_weight=sample_weight[cam_id])
                except (TypeError, ValueError):
                    # some models do not like `sample_weight` in the `fit` call...
                    # catch the exception and try again without the weights
                    self.model_dict[cam_id].fit(X[cam_id], y[cam_id])

        return self

    # def predict(self, X, cam_id=None):
    #     """
    #     In the tradition of scikit-learn, `.predict` takes a "list of feature-lists" and
    #     returns an estimate for targeted quantity for every  set of features.
    #
    #     Parameters
    #     ----------
    #     X : list of lists of floats
    #         the list of feature-lists
    #     cam_id : any (e.g. string, int), optional
    #         identifier of the singular camera type to consider here
    #         if not set, `.reg_dict` is assumed to have a single key which is used in place
    #
    #     Returns
    #     -------
    #     predict : list of floats
    #         predictions for the target quantity for every set of features given in `X`
    #
    #     Raises
    #     ------
    #     ValueError
    #         if `cam_id is None` and the number of registered models is not 1
    #     """
    #
    #     if cam_id is None:
    #         if len(self.model_dict) == 1:
    #             cam_id = next(iter(self.model_dict.keys()))
    #         else:
    #             raise ValueError("you need to provide a cam_id")
    #
    #     return self.model_dict[cam_id].predict(X)*self.energy_unit

    def save(self, path):
        """saves the models in `.reg_dict` each in a separate pickle to disk

        TODO: investigate more stable containers to write out models
        than joblib dumps

        Parameters
        ----------
        path : string
            Path to store the different models.  Expects to contain
            `{cam_id}` or at least an empty `{}` to replace it with
            the keys in `.reg_dict`.

        """

        import joblib
        for cam_id, model in self.model_dict.items():
            try:
                # assume that there is a `{cam_id}` keyword to replace
                # in the string
                joblib.dump(model, path.format(cam_id=cam_id))
            except IndexError:
                # if not, assume there is a naked `{}` somewhere left
                # if not, format won't do anything, so it doesn't
                # break but will overwrite every pickle with the
                # following one
                joblib.dump(model, path.format(cam_id))

    @classmethod
    def load(cls, path, cam_id_list, unit=1):
        """Load the pickled dictionary of model from disk, create a husk
        `cls` instance and fill the model dictionary.

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
        unit : 1 or astropy unit, optional
            scikit-learn regressor/classifier do not work with
            units. so append this one to the predictions in case you
            deal with unified targets (like energy).  assuming that
            the models where trained with consistent units.  clf
        self : RegressorClassifierBase
            in derived classes, this will return a ready-to-use
            instance of that class to predict any problem you have
            trained for

        """
        import joblib

        # need to get an instance of this class `cam_id_list=None`
        # prevents `.__init__` to initialise `.model_dict` itself,
        # since we are going to set it with the pickled models
        # manually
        self = cls(cam_id_list=None, unit=unit)
        for key in cam_id_list:
            try:
                # assume that there is a `{cam_id}` keyword to replace
                # in the string
                self.model_dict[key] = joblib.load(path.format(cam_id=key))
            except IndexError:
                # if not, assume there is a naked `{}` somewhere left
                # if not, format won't do anything, so it doesn't matter
                # though this will load the same model for every `key`
                self.model_dict[key] = joblib.load(path.format(key))

        return self

    @staticmethod
    def scale_features(cam_id_list, feature_list):
        """Scales features before training with any ML method.

        Parameters
        ----------
        cam_id_list : list
            list of camera identifiers like telescope ID or camera ID
            and the assumed distinguishing feature in the filenames of
            the various pickled regressors.
        feature_list : dictionary of lists of lists
            Dictionary that maps the telescope identifiers to lists of
            feature-lists.  The values of the dictionary are the lists
            `scikit-learn` regressors train on and are supposed to
            comply to their format requirements e.g. each feature-list
            has to contain the same features at the same position

        Returns
        -------
        f_dict : dictionary of lists of lists
            a copy of feature_list input, with scaled values

        """
        f_dict = {}
        scaler = {}

        for cam_id in cam_id_list or []:
            scaler[cam_id] = StandardScaler()
            scaler[cam_id].fit(feature_list[cam_id])
            f_dict[cam_id] = scaler[cam_id].transform(feature_list[cam_id])

        return f_dict, scaler

    def show_importances(self):
        """Creates a matplotlib figure that shows the importances of the
        different features for the various trained models in a grid of
        barplots.  The features are sorted by descending importance.

        Parameters
        ----------
        feature_labels : list of strings, optional
            a list of the feature names in proper order
            to be used as x-axis tick-labels

        Returns
        -------
        fig : matplotlib figure
            the figure holding the different bar plots

        """

        import matplotlib.pyplot as plt
        n_tel_types = len(self.model_dict)
        n_cols = np.ceil(np.sqrt(n_tel_types)).astype(int)
        n_rows = np.ceil(n_tel_types / n_cols).astype(int)

        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, squeeze=False)
        plt.suptitle("Feature Importances")
        for i, (cam_id, model) in enumerate(self.model_dict.items()):
            plt.sca(axs.ravel()[i])
            plt.title(cam_id)
            try:
                importances = model.feature_importances_
            except:
                plt.gca().axis('off')
                continue
            bins = range(importances.shape[0])

            if cam_id in self.input_features_dict \
                    and (len(self.input_features_dict[cam_id]) == len(bins)):
                feature_labels = self.input_features_dict[cam_id]
                importances, s_feature_labels = \
                    zip(*sorted(zip(importances, feature_labels), reverse=True))
                plt.xticks(bins, s_feature_labels, rotation=17)
            plt.bar(bins, importances,
                    color='r', align='center')

        # switch off superfluous axes
        for j in range(i + 1, n_rows * n_cols):
            axs.ravel()[j].axis('off')

        return fig
