"""
Functions to retrieve and interpolate atmosphere profiles.
"""
import numpy as np
from astropy.units import Quantity
from scipy.interpolate import interp1d

from ctapipe.utils import get_table_dataset

__all__ = ['get_atmosphere_profile_table', 'get_atmosphere_profile_functions']


def get_atmosphere_profile_table(atmosphere_name='paranal'):
    """
    Get an atmosphere profile table

    Parameters
    ----------
    atmosphere_name: str
        identifier of atmosphere profile

    Returns
    -------
    astropy.table.Table  containing atmosphere profile with at least columns
    'altitude' (m), and 'thickness' (g cm-2) as well as others.

    """
    table_name = f'{atmosphere_name}.atmprof'
    table = get_table_dataset(table_name=table_name,
                              role='dl0.arr.svc.atmosphere')
    return table


def get_atmosphere_profile_functions(atmosphere_name="paranal",
                                     with_units=True):
    """
    Gives atmospheric profile as a continuous function thickness(
    altitude), and it's inverse altitude(thickness)  in m and g/cm^2
    respectively.

    Parameters
    ----------
    atmosphere_name: str
        identifier of atmosphere profile
    with_units: bool
       if true, return functions that accept and return unit quantities.
       Otherwise assume units are 'm' and 'g cm-2'

    Returns
    -------
    functions: thickness(alt), alt(thickness)
    """
    tab = get_atmosphere_profile_table(atmosphere_name)
    alt = tab['altitude'].to('m')
    thick = (tab['thickness']).to("g cm-2")

    alt_to_thickness = interp1d(x=np.array(alt), y=np.array(thick))
    thickness_to_alt = interp1d(x=np.array(thick), y=np.array(alt))

    if with_units:
        def thickness(a):
            return Quantity(alt_to_thickness(a.to('m')), 'g cm-2')

        def altitude(a):
            return Quantity(thickness_to_alt(a.to('g cm-2')), 'm')

        return thickness, altitude

    return alt_to_thickness, thickness_to_alt
