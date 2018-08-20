import numpy as np


def any_array_to_numpy(any_array):
    any_array_type_to_numpy_type = {
        1: np.int8,
        2: np.uint8,
        3: np.int16,
        4: np.uint16,
        5: np.int32,
        6: np.uint32,
        7: np.int64,
        8: np.uint64,
        9: np.float,
        10: np.double,
    }
    if any_array.type == 0:
        if any_array.data:
            raise Exception("any_array has no type", any_array)
        else:
            return np.array([])
    if any_array.type == 11:
        print(any_array)
        raise Exception(
            "I have no idea if the boolean representation of"
            " the anyarray is the same as the numpy one",
            any_array
        )

    return np.frombuffer(
        any_array.data,
        any_array_type_to_numpy_type[any_array.type]
    )
