import numpy as np

from astropy import units as u

from sklearn.ensemble import RandomForestRegressor


class EnergyRegressor:
    """
    This class collects one regressor for every camera type -- given by `cam_id_list` --
    to get an estimate for the energy of an air-shower.
    The class interfaces with `scikit-learn` in that it relays all function calls not
    exeplicitly defined here through `.__getattr__` to any regressor in its collection.
    This gives us the full power and convenience of scikit-learn and we only have to
    bother about the high-level interface.

    The fitting approach is to train on single images; then separately predict an image's
    energy and combine the different predictions for all the telscopes in each event in
    one single energy  estimate.

    TODO: come up with a better merging approach to combine the estimators of the various
          camera types

    Parameters
    ----------
    regressor : scikit-learn regressor, optional (default: RandomForestRegressor)
        the regressor you want to use to estimate the shower energies
    cam_id_list : list of strings
        list of identifiers to differentiate the various sources of the images; could be
        the camera IDs or even telescope IDs. We will train one regressor for each of the
        identifiers.
    energy_unit : astropy quantity, optional (default: u.TeV)
        scikit-learn regressors don't work with astropy unit. so, tell in advance in which
        unit we want to deal here.
    kwargs
        arguments to be passed on to the constructor of the regressors
    """
    def __init__(self, regressor=RandomForestRegressor,
                 cam_id_list=("cam"), energy_unit=u.TeV, **kwargs):

        self.reg_dict = {}
        self.energy_unit = energy_unit
        for cam_id in cam_id_list:
            self.reg_dict[cam_id] = regressor(**kwargs)

    def __getattr__(self, attr):
        """
        We interface this class with the "first" regressor in `.reg_dict` and relay all
        function calls over to it. This gives us access to all the fancy implementations
        in the scikit-learn classes right from this class and we only have to overwrite
        the high-level interface functions.
        """
        return getattr(next(iter(self.reg_dict.values())), attr)

    def __str__(self):
        """ return the class name of the used regressor """
        return str(next(iter(self.reg_dict.values()))).split("(")[0]

    def reshuffle_event_list(self, X, y):
        """
        I collect the data event-wise as dictionaries of the different camera identifiers
        but for the training it is more convenient to have a dictionary containing flat
        arrays of the images. This function flattens the `X` and `y` arrays accordingly.

        This is only for convenience during the training and testing phase, later on, we
        won't collect data first event- and then telescope-wise. That's why this is a
        separate function and not implemented in `.fit`.

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
        KeyError
            in case `X` contains keys that were not provided with `cam_id_list` during
            `.__init__` or `.load`.

        Notes
        -----
        `X` and `y` are expected to be lists of events:
        `X = [event_1, event_2, ..., event_n]`
        `y = [energy_1, energy_2, ..., energy_n]`

        with `event_i` being a dictionary:
        `event_i = {tel_type_1: list_of_tels_1, ...}`
        and `energy_i` is the energy of `event_i`

        `list_of_tels_i` being the list of all telescope of type `tel_type_1` containing
        the features to fit and predict the shower energy:
        `list_of_tels_i = [[tel_1_feature_1, tel_1_feature_2, ...],
                           [tel_2_feature_1, tel_2_feature_2, ...], ...]

        after reshuffling, the resulting lists will look like this
        `trainFeatures = {tel_type_1: [features_event_1_tel_1,
                                       features_event_1_tel_2,
                                       features_event_2_tel_1, ...],
                          tel_type_2: [features_event_1_tel_3,
                                       features_event_2_tel_3,
                                       features_event_2_tel_4, ...],
                          ...}`
        `trainTarget` will be a dictionary with the same keys and a lists of energies
        corresponding to the features in the `trainFeatures` lists.
        """

        trainFeatures = {a: [] for a in self.reg_dict}
        trainTarget = {a: [] for a in self.reg_dict}

        for evt, en in zip(X, y):
            for cam_id, tels in evt.items():
                try:
                    # append the features-lists of the current event and telescope type to
                    # the flat list of features-lists for this telescope type
                    trainFeatures[cam_id] += tels
                except KeyError:
                    raise KeyError("cam_id '{}' in X but no regressor defined: {}"
                                   .format(cam_id, [k for k in self.reg_dict]))

                try:
                    # add an energy-entry for every feature-list
                    trainTarget[cam_id] += [en.to(self.energy_unit).value]*len(tels)
                except AttributeError:
                    # in case the energy is not given as an astropy quantity
                    # let's hope that the user keeps proper track of the unit themself
                    trainTarget[cam_id] += [en]*len(tels)
        return trainFeatures, trainTarget

    def fit(self, X, y):
        """
        This function fits a model against the collected features; separately for every
        telescope identifier.

        Parameters
        ----------
        X : dictionary of lists of lists
            Dictionary that maps the telescope identifiers to lists of feature-lists.
            The values of the dictionary are the lists `scikit-learn` regressors train on
            and are supposed to comply to their format requirements
            e.g. each featuer-list has to contain the same features at the same position
        y : dictionary of lists
            the energies corresponding to all the feature-lists of `X`

        Returns
        -------
        self

        Raises
        ------
        KeyError
            in case `X` contains keys that are either not in `y` were not provided before
            with `cam_id_list`.

        """
        for cam_id in X:
            if cam_id not in y:
                raise KeyError("cam_id '{}' in X but not in y: {}"
                               .format([k for k in y]))

            if cam_id not in self.reg_dict:
                raise KeyError("cam_id '{}' in X but no regressor defined: {}"
                               .format(cam_id, [k for k in self.reg_dict]))

            # for every `cam_id` train one regressor
            self.reg_dict[cam_id].fit(X[cam_id], y[cam_id])

        return self

    def predict(self, X, cam_id=None):
        """
        In the tradition of scikit-learn, `.predict` takes a "list of feature-lists" and
        returns an estimate for targeted quantity for every  set of features.

        Parameters
        ----------
        X : list of lists of floats
            the list of feature-lists
        cam_id : any (e.g. string, int), optional
            identifier of the singular camera type to consider here
            if not set, `.reg_dict` is assumed to have a single key which is used in place

        Returns
        -------
        predict : list of floats
            predictions for the target quantity for every set of features given in `X`

        Raises
        ------
        ValueError
            if `cam_id is None` and the number of registered regressors is not 1
        """

        if cam_id is None:
            if len(self.reg_dict) == 1:
                cam_id = next(iter(self.reg_dict.keys()))
            else:
                raise ValueError("you need to provide a cam_id")

        return self.reg_dict[cam_id].predict(X)*self.energy_unit

    def predict_by_event(self, X):
        """
        expects a list of events where every "event" is a dictionary mapping telescope
        identifiers to the list of feature-lists by the telescopes of that type in this
        event. Refer to the Note section in `.reshuffle_event_list` for an description how
        `X` is supposed to look like.
        The singular estimate for the event is simply the mean of the various estimators
        of the event.

        X : list of "events"
            cf. `.reshuffle_event_list` under Notes

        Returns
        -------
        dict : dictionary quantified numpy arrays
            dictionary that contains various statistical modes (mean, median, standard
            deviation) of the predicted quantity of every telescope for all events.

        Raises
        ------
        KeyError
            if there is a telescope identifier in `X` that is not a key in the regressor
            dictionary
        """

        predict_mean = []
        predict_median = []
        predict_std = []
        for evt in X:
            res = []
            for cam_id, tels in evt.items():
                try:
                    t_res = self.reg_dict[cam_id].predict(tels).tolist()
                    res += t_res
                except KeyError:
                    # QUESTION if there is no trained classifier for `cam_id`, raise an
                    # error or just pass this camera type?
                    raise KeyError("cam_id '{}' in X but no regressor defined: {}"
                                   .format(cam_id, [k for k in self.reg_dict]))

            predict_mean.append(np.mean(res))
            predict_median.append(np.median(res))
            predict_std.append(np.std(res))

        return {"mean": np.array(predict_mean)*self.energy_unit,
                "median": np.array(predict_median)*self.energy_unit,
                "std": np.array(predict_std)*self.energy_unit}

    def predict_by_telescope_type(self, X):
        """
        same as `predict_dict` only that it returns a list of dictionaries with an
        estimate for the target quantity for every telescope type separately.

        more for testing- and performance-measuring-purpouses -- to see how the different
        telescope types behave throughout the energy ranges and if a better
        (energy-dependant) combination of the separate telescope-wise estimators (compared
        to the mean) can be achieved.
        """

        predict_list_dict = []
        for evt in X:
            res_dict = {}
            for cam_id, tels in evt.items():
                t_res = self.reg_dict[cam_id].predict(tels).tolist()
                res_dict[cam_id] = np.mean(t_res)*self.energy_unit
            predict_list_dict.append(res_dict)

        return predict_list_dict

    def save(self, path):
        """
        saves the regressors in `.reg_dict` each in a separate pickle to disk

        Parameters
        ----------
        path : string
            Path to store the different regressor models as.
            Expects to contain `{cam_id}` or at least an empty `{}` to replace it with the
            keys in `.reg_dict`.

        """

        from sklearn.externals import joblib
        for cam_id, reg in self.reg_dict.items():
            try:
                # assume that there is a `{cam_id}` keyword to replace in the string
                joblib.dump(reg, path.format(cam_id=cam_id))
            except IndexError:
                # if not, assume there is a naked `{}` somewhere left
                # if not, format won't do anything, so it doesn't break but will overwrite
                # every pickle with the following one
                joblib.dump(reg, path.format(cam_id))

    @classmethod
    def load(cls, path, cam_id_list, energy_unit=u.TeV):
        """
        Load the pickled dictionary of energy regressor from disk, create a husk
        `cls` instance and fill the regressor dictionary.

        Parameters
        ----------
        path : string
            the path where the pre-trained, pickled regressors are stored
            `path` is assumed to contain a `{cam_id}` keyword to be replaced by each
            camera identifier in `cam_id_list` (or at least a naked `{}`).
        cam_id_list : list
            list of camera identifiers like telescope ID or camera ID and the assumed
            distinguishing feature in the filenames of the various pickled regressors.
        energy_unit : astropy quantity, optional (default: u.TeV)
            scikit-learn regressor do not work with units. so append this one to the
            predictions. assuming that the models where trained with consistent units.

        Returns
        -------
        self : EnergyRegressor
            a ready-to-use instance of this class to predict any quantity you have trained
            for
        """
        from sklearn.externals import joblib

        # need to get an instance of this class
        # `cam_id_list=[]` prevents `.__init__` to initialise `.reg_dict` itself,
        # since we are going to set it with the pickled models manually
        self = cls(cam_id_list=[], energy_unit=energy_unit)
        for key in cam_id_list:
            try:
                # assume that there is a `{cam_id}` keyword to replace in the string
                self.reg_dict[key] = joblib.load(path.format(cam_id=key))
            except IndexError:
                # if not, assume there is a naked `{}` somewhere left
                # if not, format won't do anything, so it doesn't matter
                # though this will load the same model for every `key`
                self.reg_dict[key] = joblib.load(path.format(key))

        return self

    def show_importances(self, feature_labels=None):
        """
        Creates a matplotlib figure that shows the importances of the different features
        for the various trained regressors in a grid of barplots.
        The features are sorted by descending importance.

        Parameters
        ----------
        feature_labels : list of strings, optional
            a list of the feature names in proper ordr
            to be used as x-axis tick-labels

        Returns
        -------
        fig : matplotlib figure
            the figure holding the different bar plots
        """
        import matplotlib.pyplot as plt
        n_tel_types = len(self.reg_dict)
        n_cols = np.ceil(np.sqrt(n_tel_types)).astype(int)
        n_rows = np.ceil(n_tel_types / n_cols).astype(int)

        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, squeeze=False)
        plt.suptitle("Feature Importances")
        for i, (cam_id, reg) in enumerate(self.reg_dict.items()):
            plt.sca(axs.ravel()[i])
            importances = reg.feature_importances_
            bins = range(importances.shape[0])
            plt.title(cam_id)
            if feature_labels:
                importances, s_feature_labels = \
                    zip(*sorted(zip(importances, feature_labels), reverse=True))
                plt.xticks(bins, s_feature_labels, rotation=17)
            plt.bar(bins, importances,
                    color='r', align='center')
        return fig
