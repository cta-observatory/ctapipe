from tempfile import TemporaryDirectory

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ctapipe.reco.event_classifier import EventClassifier


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

    clf = EventClassifier(cam_id_list=cam_id_list, n_estimators=10)
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


def test_Qfactor():
    """
    TODO: how to test validity of Q-factor values?
    """
    cam_id = ["ASTRICam"]
    features = {"ASTRICam": [[10, 1], [20, 2], [30, 3], [9, 0.9],
                             [1, 10], [2, 20], [3, 30], [0.9, 9]]}
    target = {"ASTRICam": [1, 1, 1, 1, 0, 0, 0, 0]}
    clf = EventClassifier(cam_id_list=cam_id, n_estimators=10)
    clf.fit(features, target)

    # Now predict
    ev_feat = [{"ASTRICam": [[10, 1]]}, {"ASTRICam": [[2, 20]]}, {"ASTRICam": [[3, 30]]},
               {"ASTRICam": [[100, 10]]}, {"ASTRICam": [[4, 40]]}, {"ASTRICam": [[0.5, 5]]}]
    true_labels = np.array([1, 0, 0, 1, 0, 0], dtype=np.int8)
    prediction = clf.predict_proba_by_event(X=ev_feat)

    # prediction is a two columns array
    # first  column is the probability to belong to class "0" --> hadron
    # second column is the probability to belong to class "1" --> gamma
    # we are interested in the probability to be gamma
    proba_to_be_gamma = prediction[:, 1]

    Q, gammaness = clf.compute_Qfactor(
        proba=proba_to_be_gamma, labels=true_labels, nbins=2)

    assert Q.size != 0
    assert Q.size == gammaness.size


def test_hyperBinning():
    clf = EventClassifier(cam_id_list=None)
    x = np.array([[26, 70, 53],
                  [97, 20, 56],
                  [35, 38, 81],
                  [48, 60, 40],
                  [73, 68, 63],
                  [96, 86, 63],
                  [73, 67, 6],
                  [48, 66, 60],
                  [47, 82, 87],
                  [60, 52, 74]])

    dum_l = [{'maxf': max(x[:, 0]), 'minf': min(x[:, 0]), 'col': 0, 'nbins': 4},
             {'maxf': max(x[:, 1]), 'minf': min(x[:, 1]), 'col': 1, 'nbins': 2}]

    dum_g = clf._hyperBinning(x, dum_l)

    assert np.all(dum_g.size() == (1, 1, 3, 2, 1))


def test_level_populations():
    clf = EventClassifier(cam_id_list=None)
    g = np.array([[26, 70, 53],
                  [97, 20, 56],
                  [35, 38, 81],
                  [48, 60, 40],
                  [73, 68, 63],
                  [96, 86, 63],
                  [73, 67, 6],
                  [48, 66, 60],
                  [47, 82, 87],
                  [60, 52, 74]])

    h = np.array([[18, 31, 47],
                  [15, 81, 72],
                  [75, 93, 45],
                  [57, 50, 3],
                  [12, 80, 3],
                  [82, 49, 31],
                  [1, 21, 0],
                  [79, 12, 29],
                  [19, 52, 42],
                  [86, 49, 15]])

    dum_l = [{'maxf': 100, 'minf': 0, 'col': 0, 'nbins': 4},
             {'maxf': 100, 'minf': 0, 'col': 1, 'nbins': 2}]

    group_g = clf._hyperBinning(g, dum_l)
    group_h = clf._hyperBinning(h, dum_l)

    cleaned_g, cleaned_h = clf.level_populations(group_g, group_h, g, h)

    assert cleaned_g.shape == cleaned_h.shape


