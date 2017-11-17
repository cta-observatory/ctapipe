# Licensed under a 3-clause BSD style license - see LICENSE.rst
from .fitshistogram import Histogram
from .json2fits import json_to_fits
from .dynamic_class import dynamic_class_from_module
from .table_interpolator import TableInterpolator
from .datasets import (find_all_matching_datasets, get_table_dataset, get_dataset,
                       find_in_path)
from .coordinate_transformations import (guess_pix_direction, alt_to_theta, az_to_phi,
                                         transf_array_position, transf_pixel_position)

# indices to play through the different transformations
# set variables to control which transformation to use
# e.g. loop over the indices to bruteforce the correct set of transformations
coordinate_transformations.pixel = 2
coordinate_transformations.array = 0
coordinate_transformations.azimu = 0
# this is supposed to be +1 or -1 to set the direction the azimuth is turning
coordinate_transformations.az_dir = -1
