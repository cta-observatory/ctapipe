#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read an atmospheric profile (Density, P, T...)

@author: Johan Bregeon
"""

import logging
logger = logging.getLogger(__name__)


ATMOPROF_DICT = {'Armazones': 26, 'La Palma': 36}


def read_atmo_prof(filepath):
    """ Read atmospheric profile table

    Parameters
    ----------
    filepath : atmospheric profile file name, shall be available
               as a ctapipe "dataset"

    Returns
    ----------
    7 lists corresponding to:
    altitude, rho, thick, index, temperature, pressure, pw_p

    """
    content = open(filepath).readlines()

    altitude = []
    rho = []
    thick = []
    index = []
    temperature = []
    pressure = []
    pw_p = []
    for line in content:
        if line[0] is '#':
            logger.info(line)
        else:
            # Alt [km]    rho [g/cm^3] thick [g/cm^2]    n-1        T [K]\
            # p [mbar]      pw / p
            all_el = [float(el) for el in line.split()]
            altitude.append(all_el[0] * 1000.)
            rho.append(all_el[1])
            thick.append(all_el[2])
            index.append(all_el[3])
            temperature.append(all_el[4])
            pressure.append(all_el[5])
            pw_p.append(all_el[6])
    return altitude, rho, thick, index, temperature, pressure, pw_p


if __name__ == "__main__":
    """ Open and read atmoprof26.dat to get its content in forms of lists
        @TODO:
            * print a few values
    """
    from traitlets import Unicode
    from ctapipe.utils import get_dataset
    # set CTAPIPE_SVC_PATH to a path containing the file below
    input_path = get_dataset('atmprof26.dat')

    # JB can't make this work...
    # input_path = Unicode(get_dataset('atmprof26.dat'), allow_none=True,
    #                     help='Path to the atmospheric profile file, e.g. '
    #                          'atmprof26.dat').tag(config=True)
    altitudes, rhos, thicks, indexs, temperatures, pressures, pw_ps = \
        read_atmo_prof(input_path)
