from tempfile import TemporaryDirectory
from ctapipe.reco.event_classifier import *
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pytest


def test_pipeline_classifier():
    cam_id_list = ["FlashCam", "ASTRICam"]
    feature_list = {"FlashCam": [[1, 10], [2, 20], [3, 30], [0.9, 9],
                                 [10, 1], [20, 2], [30, 3], [9, 0.9]],
                    "ASTRICam": [[10, 1], [20, 2], [30, 3], [9, 0.9],
                                 [1, 10], [2, 20], [3, 30], [0.9, 9]]}
    target_list = {"FlashCam": ["a", "a", "a", "a", "b", "b", "b", "b"],
                   "ASTRICam": ["a", "a", "a", "a", "b", "b", "b", "b"]}

    estimators = [('scaler', StandardScaler()),
                  ('clf', MLPClassifier(max_iter=400))]

    clf = EventClassifier(classifier=Pipeline, steps=estimators,
                          cam_id_list=cam_id_list)
    clf.fit(feature_list, target_list)

    prediction = clf.predict_by_event([{"ASTRICam": [[10, 1]]},
                                       {"ASTRICam": [[2, 20]]},
                                       {"ASTRICam": [[3, 30]]}])
    assert (prediction == ["a", "b", "b"]).all()

    prediction = clf.predict_by_event([{"FlashCam": [[10, 1]]},
                                       {"FlashCam": [[2, 20]]},
                                       {"FlashCam": [[3, 30]]}])
    assert (prediction == ["b", "a", "a"]).all()


def test_prepare_model_MLP():
    cam_id_list = ["FlashCam", "ASTRICam"]
    feature_list = {"FlashCam": [[1, 10], [2, 20], [3, 30], [0.9, 9],
                                 [10, 1], [20, 2], [30, 3], [9, 0.9]],
                    "ASTRICam": [[10, 1], [20, 2], [30, 3], [9, 0.9],
                                 [1, 10], [2, 20], [3, 30], [0.9, 9]]}
    target_list = {"FlashCam": ["a", "a", "a", "a", "b", "b", "b", "b"],
                   "ASTRICam": ["a", "a", "a", "a", "b", "b", "b", "b"]}

    clf = EventClassifier(classifier=MLPClassifier, cam_id_list=cam_id_list, max_iter=400)
    scaled_features, scaler = EventClassifier.scale_features(cam_id_list, feature_list)

    # clf.fit(feature_list, target_list)
    clf.fit(scaled_features, target_list)
    return clf, cam_id_list, scaler

def test_fit_save_load_MLP():
    clf, cam_id_list, scaler = test_prepare_model_MLP()
    with TemporaryDirectory() as d:
        temp_path = "/".join([d, "reg_{cam_id}.pkl"])
        clf.save(temp_path)
        clf = EventClassifier.load(temp_path, cam_id_list)
        return clf, cam_id_list, scaler

def test_predict_by_event_MLP():
    clf, cam_id_list, scaler = test_fit_save_load_MLP()

    x = scaler['ASTRICam'].transform(np.array([10, 1], dtype=float).reshape(1, -1))
    x = {'ASTRICam': x}
    y = scaler['ASTRICam'].transform(np.array([2, 20], dtype=float).reshape(1, -1))
    y = {'ASTRICam': y}
    z = scaler['ASTRICam'].transform(np.array([3, 30], dtype=float).reshape(1, -1))
    z = {'ASTRICam': z}
    prediction = clf.predict_by_event([x, y, z])
    assert (prediction == ["a", "b", "b"]).all()

    x = scaler['FlashCam'].transform(np.array([10, 1], dtype=float).reshape(1, -1))
    x = {'FlashCam': x}
    y = scaler['FlashCam'].transform(np.array([2, 20], dtype=float).reshape(1, -1))
    y = {'FlashCam': y}
    z = scaler['FlashCam'].transform(np.array([3, 30], dtype=float).reshape(1, -1))
    z = {'FlashCam': z}
    prediction = clf.predict_by_event([x, y, z])
    assert (prediction == ["b", "a", "a"]).all()

def test_prepare_model():
    cam_id_list = ["FlashCam", "ASTRICam"]
    feature_list = {"FlashCam": [[1, 10], [2, 20], [3, 30], [0.9, 9],
                                 [10, 1], [20, 2], [30, 3], [9, 0.9]],
                    "ASTRICam": [[10, 1], [20, 2], [30, 3], [9, 0.9],
                                 [1, 10], [2, 20], [3, 30], [0.9, 9]]}
    target_list = {"FlashCam": ["a", "a", "a", "a", "b", "b", "b", "b"],
                   "ASTRICam": ["a", "a", "a", "a", "b", "b", "b", "b"]}

    clf = EventClassifier(cam_id_list=cam_id_list)
    clf.fit(feature_list, target_list)
    return clf, cam_id_list

def test_fit_save_load():
    clf, cam_id_list = test_prepare_model()
    with TemporaryDirectory() as d:
        temp_path = "/".join([d, "reg_{cam_id}.pkl"])
        clf.save(temp_path)
        clf = EventClassifier.load(temp_path, cam_id_list)
        return clf, cam_id_list


def test_predict_by_event():
    clf, cam_id_list = test_fit_save_load()
    prediction = clf.predict_by_event([{"ASTRICam": [[10, 1]]},
                                       {"ASTRICam": [[2, 20]]},
                                       {"ASTRICam": [[3, 30]]}])
    assert (prediction == ["a", "b", "b"]).all()

    prediction = clf.predict_by_event([{"FlashCam": [[10, 1]]},
                                       {"FlashCam": [[2, 20]]},
                                       {"FlashCam": [[3, 30]]}])
    assert (prediction == ["b", "a", "a"]).all()
