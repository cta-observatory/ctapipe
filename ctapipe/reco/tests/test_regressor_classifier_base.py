import pytest
from sklearn.ensemble import RandomForestClassifier
from ctapipe.reco.regressor_classifier_base import RegressorClassifierBase


def test_reshuffle_event_list():
    feature_list = [
        {"FlashCam": [[1, 10], [2, 20]],
         "ASTRICam": [[30, 3], [40, 4]]},
        {"FlashCam": [[1.5, 15]],
         "ASTRICam": [[35, 3.5], [25, 2.5], [30, 3]]}
    ]
    target_list = ["1", "2"]

    cam_id_list = ["FlashCam", "ASTRICam"]
    my_base = RegressorClassifierBase(model=RandomForestClassifier,
                                      cam_id_list=cam_id_list, unit=1, n_estimators=100)

    feature_flattened, target_flattened = my_base.reshuffle_event_list(
        feature_list, target_list
    )

    assert feature_flattened == {'FlashCam': [[1, 10], [2, 20], [1.5, 15]],
                                 'ASTRICam': [[30, 3], [40, 4], [35, 3.5],
                                              [25, 2.5],
                                              [30, 3]]}

    assert target_flattened == {'FlashCam': ['1', '1', '2'],
                                'ASTRICam': ['1', '1', '2', '2', '2']}

    assert len(str(my_base)) > 0


def test_failures():
    cam_id_list = ["FlashCam", "ASTRICam"]
    my_base = RegressorClassifierBase(model=RandomForestClassifier,
                                      cam_id_list=cam_id_list, unit=1)

    # some test data with a bad camera in it should raise a KeyError
    feature_list = [
        {"FlashCam": [[1, 10], [2, 20]],
         "BadCam": [[30, 3], [40, 4]]},
        {"FlashCam": [[1.5, 15]],
         "ASTRICam": [[35, 3.5], [25, 2.5], [30, 3]]}
    ]
    target_list = ["1", "2"]

    with pytest.raises(KeyError):
        feature_flattened, target_flattened = my_base.reshuffle_event_list(
            feature_list,
            target_list
        )
        assert feature_flattened is not None
        assert target_flattened is not None


def test_show_importances():

    cam_id_list = ["FlashCam", "ASTRICam"]
    feature_list = {"FlashCam": [[1, 10], [2, 20], [3, 30], [0.9, 9],
                                 ],
                    "ASTRICam": [[10, 1], [20, 2], [30, 3], [9, 0.9],
                                 ]}
    target_list = {"FlashCam": [0, 1, 1, 0],
                   "ASTRICam": [1, 0, 0, 0]}

    reg = RegressorClassifierBase(
        model=RandomForestClassifier,
        cam_id_list=cam_id_list,
        unit=1,
        n_estimators=10,
    )

    reg.fit(feature_list, target_list)
    reg.input_features_dict = {
        "FlashCam": ['f1', 'f2'],
        "ASTRICam": ['f1', 'f2'],
    }
    fig = reg.show_importances()
    ax = fig.axes[0]
    assert len(ax.get_xticklabels()) == 2
    for t in ax.get_xticklabels():
        assert t.get_text() in ['f1', 'f2']
