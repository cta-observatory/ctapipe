import sys
"""
This is a template for the camera prototypes for
a uniform implementation of the different camera calibration
functions.

WE ENCOURAGE to follow the current template together with the
coding guidelines document:

FUNCTIONS: There will be two types of functions:
- <calibration_constant_name>_calculation(): functions that calculates
calibration constants.
- <calibration_constant_name>_application(): functions that applies
the calculated calibration constant to the raw data for its calibration.
- <main function>: a detailed scheme of how and the order of
how the different calibration functions should be called.

IMPORTANT: if the current scheme, for any reason does not fit into
your camera calibration, please contact DATA experts to search for
common alternatives.

"""

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def calibration_constant_calculation(raw_dataset, parameters):
    """

    Here would go the function that calculates your pedestal during
    the observations

    """
    return nsb_pedestals


def pedestal_substraction(raw_dataset, calibration_dataset, parameters):
    """
    Here would go the function that substracts the pedestal to your raw data
    """
    return net_raw_pixel_amplitude


if __name__ == "__main__":

    if (len(sys.argv) < 1):
        logger.info("Not enough arguments. Usage %s " % sys.argv[0])
        sys.exit(0)

    calibration_dataset = calibration_constant_calculation(
        raw_dataset, parameters)
    calibrated_dataset = calibration_constant_application(
        raw_dataset, calibration_dataset, parameters)

    logger.info("%s> Closing file...")
    sys.exit(0)
