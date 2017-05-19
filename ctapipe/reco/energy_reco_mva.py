import sklearn
from ctapipe.coordinates import (TiltedGroundFrame,
                                 GroundFrame)

from ctapipe.io.containers import (ReconstructedEnergyContainer)
from ctapipe.reco.reco_algorithms import EnergyReconstructor
from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler

from os.path import isfile
import numpy as np
import astropy.units as u

__all__ = ['EnergyReconstructorMVA']


class EnergyReconstructorMVA(EnergyReconstructor):
    """
    """

    def __init__(self, root_dir="."):
        """
        
        Parameters
        ----------
        root_dir
        """
        # First we create a dictionary of image template interpolators
        # for each telescope type
        self.root_dir = root_dir

        self.file_names = {"GATE": "GCT_energy.pkl", "LSTCam":
                           "LST_energy.pkl", "NectarCam":
                           "NectarCam_energy.pkl", "FlashCam": "FlashCam_energy.pkl"}
        self.input_scaler = dict()
        self.energy_estimator = dict()
        self.error_estimator = dict()

        self.type = None
        # We also need telescope positions
        self.tel_pos_x = 0
        self.tel_pos_y = 0
        self.hillas = None

        self.tilted_frame = 0

    def initialise_mva(self, tel_type):
        """
        
        Parameters
        ----------
        tel_type

        Returns
        -------

        """
        filename = self.root_dir + self.file_names[tel_type]

        if isfile(filename):
            estimator = joblib.load(filename)
            self.energy_estimator[tel_type] = estimator["regression"]
            if "error" in estimator:
                self.error_estimator[tel_type] = estimator["error"]
            if "scale" in estimator:
                self.input_scaler[tel_type] = estimator["scale"]
        else:
            print("OK MVA does not exist, so lets make it")
            gamma_list = self.root_dir + "gamma_events.fits"
            mva, scaler, error_mva = \
                train_energy_mva(gamma_list,tel_type,
                                 mva=GradientBoostingRegressor(n_estimators=100,
                                                               learning_rate=0.1,
                                                               max_depth=10,
                                                               random_state=0, loss='ls',
                                                               verbose=True),
                                 scaler = RobustScaler(),
                                 error_mva=GradientBoostingRegressor(n_estimators=100,
                                                                     learning_rate=0.1,
                                                                     max_depth=10,
                                                                     random_state=0,
                                                                     loss='ls',
                                                                     verbose=True))
            joblib_output = {"regression":mva}
            self.energy_estimator[tel_type] = mva

            if scaler is not None:
                joblib_output["scaler"] = scaler
                self.input_scaler[tel_type] = scaler
            if error_mva is not None:
                joblib_output["error"] = error_mva
                self.error_estimator[tel_type] = error_mva

            joblib.dump(joblib_output, filename)

    def estimate_energy(self, core):
        """
        
        Parameters
        ----------
        core

        Returns
        -------

        """
        tel_number = len(self.tel_pos_x)
        energy = np.zeros(tel_number)
        error = np.ones(tel_number)

        for tel, num in zip(self.tel_pos_x, range(tel_number)):

            impact = np.sqrt(np.power(self.tel_pos_x[tel] - core.x, 2) +
                             np.power(self.tel_pos_y[tel] - core.y, 2))

            hill = self.hillas[tel]

            tel_type = self.type[tel]
            mva_input = (hill.size, impact.to(u.m).value,
                         hill.width.to(u.rad).value,
                         hill.length.to(u.rad).value)

            if tel_type in self.input_scaler:
                mva_input = self.input_scaler[tel_type].transform(mva_input)

            energy[num] = self.energy_estimator[tel_type].predict(mva_input)

            if tel_type in self.error_estimator:
                error[num] = self.error_estimator[tel_type].predict(mva_input)

        weight = 1./error

        return np.sum(energy*weight)/np.sum(weight)

    def set_event_properties(self, hillas, type_tel, tel_x, tel_y, tilted_frame):

        """
        
        Parameters
        ----------
        hillas
        type_tel
        tel_x
        tel_y
        tilted_frame

        Returns
        -------

        """

        # First store these parameters in the class so we can use them
        # in reconstruction For most values this is simply copying

        self.tel_pos_x = tel_x
        self.tel_pos_y = tel_y
        self.hillas = hillas
        self.type = type_tel

        self.initialise_mva(type_tel)

        self.tilted_frame = tilted_frame

    def predict(self, shower):
        """
        
        Parameters
        ----------
        shower

        Returns
        -------

        """

        ground = GroundFrame(x=shower.core_x,
                             y=shower.core_y, z=0 * u.m)
        tilted = ground.transform_to(
            TiltedGroundFrame(pointing_direction=self.array_direction)
        )

        energy = self.energy_estimator(tilted)


from astropy.io import fits
from sklearn.model_selection import train_test_split

def train_energy_mva(filename, tel_type, mva, error_mva=None, scaler=None):
    """
    
    Parameters
    ----------
    filename
    tel_type
    mva
    error_mva
    scaler

    Returns
    -------

    """

    gam = fits.open(filename)
    data = gam[1].data

    mask = data['TEL_TYPE'] == tel_type[0]

    target = data[mask]["SIM_EN"]
    target = np.log10(target)

    amp = data[mask]["AMP"]
    impact = data[mask]["IMPACT"]
    width = data[mask]["WIDTH"]
    length = data[mask]["LENGTH"]

    input_vals = np.array((amp, impact, width, length))
    input_vals = input_vals.T

    return fit_mva(input_vals, tel_type, mva, error_mva, scaler)


def fit_mva(input_vals, target, mva, error_mva, scaler):
    """
    
    Parameters
    ----------
    input_vals
    target
    mva
    error_mva
    scaler

    Returns
    -------

    """

    X_train, X_test, y_train, y_test = train_test_split(input_vals, target, test_size=0.33,
                                                        random_state=42)

    if scaler is not None:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    mva.fit(X_train, y_train)
    predicted_energy = mva.predict(X_test, y_test)

    if error_mva is not None:
        bias = pow(10, predicted_energy) - pow(10, y_train)
        bias /= pow(10, y_train)

        error_mva.fit(X_test, bias)

    return mva, scaler, error_mva