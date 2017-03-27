# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Hillas shower parametrization.



"""
import numpy as np
import itertools
import astropy.units as u

__all__ = [
    'reconstruct_nominal',
    'reconstruct_tilted'
]


def reconstruct_nominal(hillas_parameters,weighting="Konrad"):
    """
    Perform event reconstruction by simple Hillas parameter intersection
    in the nominal system

    Parameters
    ----------
    hillas_parameters: Llist
        Hillas parameter objects
    weighting: string
        Specify image weighting scheme used (HESS or Konrad style)

    Returns
    -------
    Reconstructed event position in the nominal system

    """
    if len(hillas_parameters)<2:
        return None # Throw away events with < 2 images

    print(type(list(hillas_parameters.values())))

    # Find all pairs of Hillas parameters
    hillas_pairs = list(itertools.combinations(list(hillas_parameters.values()), 2))

    # Copy parameters we need to a numpy array to speed things up
    h1 = list(map(lambda h:[h[0].psi.to(u.rad).value,h[0].cen_x.value,h[0].cen_y.value,h[0].size],hillas_pairs))
    h1 = np.array(h1)
    h1 = np.transpose(h1)

    h2 = np.array(list(map(lambda h:[h[1].psi.to(u.rad).value,h[1].cen_x.value,h[1].cen_y.value,h[1].size],hillas_pairs)))
    h2 = np.array(h2)
    h2 = np.transpose(h2)

    # Perform intersection
    sx,sy = intersect_lines(h1[1],h1[2],h1[0],
                            h2[1],h2[2],h2[0])
    if weighting == "Konrad":
        weight_fn = weight_konrad
    elif weighting == "HESS":
        weight_fn = weight_HESS

    weight = weight_fn(h1[3],h2[3])
    weight *= weight_sin(h1[0],h2[0])

    # Make weighted average of all possible pairs
    sum_x = np.sum(sx*weight)
    sum_y = np.sum(sy*weight)
    sum_w = np.sum(weight)

    # Copy into nominal coordinate
    #nominal = NominalFrame(x=(sum_x/sum_w)*u.deg,y=(sum_y/sum_w)*u.deg,z=0*u.deg,array_direction=c_nom.array_direction)
    return 57.3*sum_x/sum_w,57.3*sum_y/sum_w


def reconstruct_tilted(hillas_parameters,tel_x,tel_y,weighting="Konrad"):

    if len(hillas_parameters)<2:
        return None # Throw away events with < 2 images

    # Find all pairs of Hillas parameters
    hillas_pairs = list(itertools.combinations(hillas_parameters.values, 2))
    tel_x = np.array(list(itertools.combinations(tel_x, 2)))
    tel_y= np.array(list(itertools.combinations(tel_y, 2)))

     # Copy parameters we need to a numpy array to speed things up
    h1 = list(map(lambda h:[h[0].psi.to(u.rad).value,h[0].size],hillas_pairs))
    h1 = np.array(h1)
    h1 = np.transpose(h1)

    h2 = np.array(list(map(lambda h:[h[1].psi.to(u.rad).value,h[1].size],hillas_pairs)))
    h2 = np.array(h2)
    h2 = np.transpose(h2)
    # Perform intersection
    cx,cy = intersect_lines(tel_x[:,0],tel_y[:,0],h1[0],
                            tel_x[:,1],tel_y[:,1],h2[0])

    if weighting == "Konrad":
        weight_fn = weight_konrad
    elif weighting == "HESS":
        weight_fn = weight_HESS

    weight = weight_fn(h1[1],h2[1])
    weight *= weight_sin(h1[0],h2[0])

    # Make weighted average of all possible pairs
    sum_x = np.sum(cx*weight)
    sum_y = np.sum(cy*weight)
    sum_w = np.sum(weight)

    tilt = TiltedGroundFrame(x=(sum_x/sum_w)*u.m,y=(sum_y/sum_w)*u.m,z=0*u.m,pointing_direction=[70*u.deg,0*u.deg])
    tilt.transform_to(GroundFrame)

    return tilt


def intersect_lines(xp1,yp1,phi1,xp2,yp2,phi2):
    """

    Parameters
    ----------
    xp1: ndarray
        X position of first image
    yp1: ndarray
        Y position of first image
    phi1: ndarray
        Rotation angle of first image
    xp2: ndarray
        X position of second image
    yp2: ndarray
        Y position of second image
    phi2: ndarray
        Rotation angle of second image

    Returns
    -------
    ndarray of x and y crossing points for all pairs
    """
    s1 = np.sin(phi1)
    c1 = np.cos(phi1)
    A1 = s1
    B1 = -1*c1
    C1 = yp1*c1 - xp1*s1

    s2 = np.sin(phi2)
    c2 = np.cos(phi2)

    A2 = s2
    B2 = -1*c2
    C2 = yp2*c2 - xp2*s2

    detAB = (A1*B2-A2*B1)
    detBC = (B1*C2-B2*C1)
    detCA = (C1*A2-C2*A1)

    #if  math.fabs(detAB) < 1e-14 : # /* parallel */
    #    return 0,0
    xs = detBC / detAB
    ys = detCA / detAB

    return xs,ys


def weight_konrad(p1,p2):
    return (p1*p2)/(p1+p2)


def weight_HESS(p1,p2):
    return 1/((1/p1)+(1/p2))


def weight_sin(phi1,phi2):
    return np.abs(np.sin(np.fabs(phi1-phi2)))