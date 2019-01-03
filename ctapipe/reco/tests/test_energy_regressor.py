from tempfile import TemporaryDirectory

import numpy as np
from numpy.testing import assert_allclose

from astropy import units as u

from ctapipe.reco.energy_regressor import EnergyRegressor


def test_prepare_model():
    cam_id_list = ["FlashCam", "ASTRICam"]
    feature_list = {"FlashCam": [[1, 10], [2, 20], [3, 30], [0.9, 9],
                                 ],
                    "ASTRICam": [[10, 1], [20, 2], [30, 3], [9, 0.9],
                                 ]}
    target_list = {"FlashCam": np.array([1, 2, 3, 0.9]) * u.TeV,
                   "ASTRICam": np.array([1, 2, 3, 0.9]) * u.TeV}

    reg = EnergyRegressor(cam_id_list=cam_id_list, n_estimators=10)
    reg.fit(feature_list, target_list)
    return reg, cam_id_list


def test_fit_save_load():
    reg, cam_id_list = test_prepare_model()
    with TemporaryDirectory() as d:
        temp_path = "/".join([d, "reg_{cam_id}.pkl"])
        reg.save(temp_path)
        reg = EnergyRegressor.load(temp_path, cam_id_list)
        return reg, cam_id_list


def test_predict_by_event():
    np.random.seed(3)

    reg, cam_id_list = test_fit_save_load()
    prediction = reg.predict_by_event([{"ASTRICam": [[10, 1]]},
                                       {"ASTRICam": [[20, 2]]},
                                       {"ASTRICam": [[30, 3]]}])
    assert_allclose(prediction["mean"].value, [1, 2, 3], rtol=0.2)

    prediction = reg.predict_by_event([{"FlashCam": [[1, 10]]},
                                       {"FlashCam": [[2, 20]]},
                                       {"FlashCam": [[3, 30]]}])
    assert_allclose(prediction["mean"].value, [1, 2, 3], rtol=0.2)
