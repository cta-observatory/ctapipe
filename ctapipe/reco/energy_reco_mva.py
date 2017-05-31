"""

"""
from os.path import isfile

import astropy.units as u
import numpy as np
from astropy.io import fits
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from ctapipe.coordinates import (TiltedGroundFrame,
                                 GroundFrame)
from ctapipe.io.containers import (ReconstructedEnergyContainer)
from ctapipe.reco.reco_algorithms import EnergyReconstructor

__all__ = ['EnergyReconstructorMVA']


class EnergyReconstructorMVA(EnergyReconstructor):
    """
    Class for energy reconstruction using Hillas parameters using multi variate 
    analysis. Class loads MVA from a pickle file, or if not present loads fits training 
    data and trains MVA (saving it to a pickle file when complete). There are 3 steps 
    present in MVA energy estimation:
    
    - Scaler
        Scales input variables to a defined range
    - Energy estimator
        Performs energy estimation for a given telescope type using MVA
    - Error estimator
        Estimate error on energy estimation, for use with telescope estimation weighting  
    """

    def __init__(self, root_dir="./"):
        """
        Parameters
        ----------
        root_dir: str
            Base directory containing pickle files of training sets
        """
        # First we create a dictionary of image template interpolators
        # for each telescope type
        self.root_dir = root_dir

        self.file_names = {"GCT": "GCT_energy.pkl", "LSTCam":
                           "LST_energy.pkl", "NectarCam":
                           "NectarCam_energy.pkl", "FlashCam": "FlashCam_energy.pkl"}

        # Create dictionaries for each step of energy estimation (by telescope type)
        self.input_scaler = dict()
        self.energy_estimator = dict()
        self.error_estimator = dict()

        self.type = None
        # We also need telescope positions
        self.tel_pos_x = None
        self.tel_pos_y = None
        # Hillas parameters
        self.hillas = None

        # Tilted reference frame in which to calculate impact distance
        self.tilted_frame = 0

    def initialise_mva(self, tel_type):
        """
        Initialise MVA for each telescope type in the event from a saved pickle file or 
        retrain it from training input
        
        Parameters
        ----------
        tel_type: dict
            Dictionary of telescope types in event

        Returns
        -------
            None
        """
        # Loop over telescope types
        for tel_num in tel_type:
            # If we have already loaded the tel type skip it
            if tel_type[tel_num] in self.energy_estimator:
                continue
            filename = self.root_dir + self.file_names[tel_type[tel_num]]

            # Then check to see if the pcikle file already exists
            if isfile(filename):
                # If so load it
                estimator = joblib.load(filename)
                self.energy_estimator[tel_type[tel_num]] = estimator["regression"]
                if "error" in estimator:
                    self.error_estimator[tel_type[tel_num]] = estimator["error"]
                if "scale" in estimator:
                    self.input_scaler[tel_type[tel_num]] = estimator["scale"]
            else:
                # If the MVA does not exist we load up the training set and use it to
                # train the required MVA
                gamma_list = self.root_dir + "gamma.fits"
                mva, scaler, error_mva = \
                   train_energy_mva(gamma_list,tel_type[tel_num],
                                    mva=GradientBoostingRegressor(n_estimators=100,
                                                                  learning_rate=0.1,
                                                                  max_depth=10,
                                                                  random_state=0,
                                                                  loss='ls',
                                                                  verbose=True),
                                    scaler=RobustScaler(),
                                    error_mva=GradientBoostingRegressor(n_estimators=100,
                                                                        learning_rate=0.1,
                                                                        max_depth=10,
                                                                        random_state=0,
                                                                        loss='ls',
                                                                        verbose=True))
                # Fill dictionary
                joblib_output = {"regression":mva}
                self.energy_estimator[tel_type[tel_num]] = mva

                if scaler is not None:
                    joblib_output["scaler"] = scaler
                    self.input_scaler[tel_type[tel_num]] = scaler
                if error_mva is not None:
                    joblib_output["error"] = error_mva
                    self.error_estimator[tel_type[tel_num]] = error_mva
                # Save trained MVA in a pickle file
                joblib.dump(joblib_output, filename)

    def estimate_energy(self, core):
        """
        Estimate the energy of an event at a given core position using the MVA
        
        Parameters
        ----------
        core: TiltedGroundFrame
            Core position in tilted frame

        Returns
        -------
            Energy Estimation, Spread of energy estimates
        """
        tel_number = len(self.tel_pos_x)
        energy = np.zeros(tel_number)
        error = np.ones(tel_number)

        # 2D input is required to MVA
        mva_input = np.zeros((1,4))

        # Loop over telescopes
        for tel, num in zip(self.tel_pos_x, range(tel_number)):
            # Calculate impact distance of telescope to the core
            impact = np.sqrt(np.power(self.tel_pos_x[tel] - core.x, 2) +
                             np.power(self.tel_pos_y[tel] - core.y, 2))

            hill = self.hillas[tel]

            tel_type = self.type[tel]
            # Create input to MVA
            mva_input[0] = np.array((hill.size, impact.to(u.m).value,
                                    hill.width.to(u.rad).value,
                                    hill.length.to(u.rad).value))

            # Scale MVA if we have a scaler
            if tel_type in self.input_scaler:
                mva_input = self.input_scaler[tel_type].transform(mva_input)
            # Estimate energy
            energy[num] = np.power(10,self.energy_estimator[tel_type].predict(mva_input))
            # Estimate error if we can
            if tel_type in self.error_estimator:
                error[num] = self.error_estimator[tel_type].predict(mva_input)
        # Weight by  estimate of squared error
        weight = 1./error
        average_energy = np.average(energy, weights=weight)
        variance = np.average((energy - average_energy) ** 2, weights=weight)

        return average_energy * u.TeV, np.sqrt(variance) * u.TeV

    def predict(self, shower, hillas, type_tel, tel_x, tel_y, tilted_frame):
        """
        Estimate shower energy for an event
        
        Parameters
        ----------
        shower: ReconstructedShowerContainer
            Reconstructed shower parameter
        hillas: dict
            Dictionary of telescope Hillas parameters
        type_tel: dict
            Dictionary of telescope types
        tel_x: dict
            Dictionary of telescope X positions in the tilted system
        tel_y: dict
            Dictionary of telescope Y positions in the tilted system
        tilted_frame: TiltedGroundFrame
            TiltedGroundFrame in which to perform reconstruction
            
        Returns
        -------
        ReconstructedEnergyContainer
        """

        #Store required info as class members
        self.tel_pos_x = tel_x
        self.tel_pos_y = tel_y
        self.hillas = hillas
        self.type = type_tel

        # Initialise any unloaded MVAs
        self.initialise_mva(type_tel)

        self.tilted_frame = tilted_frame

        # Convert shower core to the tiled system
        ground = GroundFrame(x=shower.core_x,
                             y=shower.core_y, z=0 * u.m)
        tilted = ground.transform_to(
            TiltedGroundFrame(pointing_direction=self.tilted_frame)
        )

        # Estimate energy and store in ReconstructedEnergyContainer
        reconstructed_energy, energy_error = self.estimate_energy(tilted)
        energy_result = ReconstructedEnergyContainer()
        energy_result.is_valid = True
        energy_result.energy = reconstructed_energy
        energy_result.energy_uncert = energy_error

        return energy_result


# If we don't have a pickled MVA we need to train using the training input
def train_energy_mva(filename, tel_type, mva, error_mva=None, scaler=None):
    """
    Load data fits training file and pass on to functions for regressor training
    
    Parameters
    ----------
    filename: str
        Training file input
    tel_type: str
        Name of telescope type
    mva: RegressorMixin
        Regressor to be trained for energy estimation
    error_mva: RegressorMixin
        Regressor to be trained for error estimation
    scaler: RobustScaler
        Scikit learn scaler, for scaling results

    Returns
    -------
    mva, scaler, error_mva
    """

    # Open gamma ray FITS training file
    gam = fits.open(filename)
    data = gam[1].data

    mask = data['TEL_TYPE'] == tel_type[0]

    # Load target energy fromfile
    target = data[mask]["SIM_EN"]
    target = np.log10(target)

    # Load Training data from file
    amp = data[mask]["AMP"]
    impact = data[mask]["IMPACT"]
    width = data[mask]["WIDTH"]
    length = data[mask]["LENGTH"]

    input_vals = np.array((amp, impact, width, length))
    input_vals = input_vals.T

    return fit_mva(input_vals, target, mva, error_mva, scaler)


def fit_mva(input_vals, target, mva, error_mva, scaler):
    """
    Perform MVA training and return trained regressor
    
    Parameters
    ----------
    input_vals: ndarray
        2D array of input variables for training
    target: ndarray
        1D array of target energies for the MVA training
    mva:  RegressorMixin
        Regressor to be trained for energy estimation
    error_mva: RegressorMixin
        Regressor to be trained for error estimation
    scaler: RobustScaler
        Scikit learn scaler, for scaling results

    Returns
    -------
    mva, scaler, error_mva

    """

    X_train, X_test, y_train, y_test = train_test_split(input_vals, target, test_size=0.33,
                                                        random_state=42)

    if scaler is not None:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    mva.fit(X_train, y_train)
    predicted_energy = mva.predict(X_test, y_test)

    # Train based of squared deviation from true values
    if error_mva is not None:
        bias = pow(10, predicted_energy) - pow(10, y_train)
        bias /= pow(10, y_train)

        error_mva.fit(X_test, bias*bias)

    return mva, scaler, error_mva