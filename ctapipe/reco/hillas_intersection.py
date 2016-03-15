# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Hillas shower parametrization.



"""
import numpy as np
import itertools

__all__ = [
    'reconstruct_nominal',
    'reconstruct_tilted'
]


def reconstruct_nominal(hillas_parameters,tel_config,weighting="Konrad"):
    if len(hillas_parameters)<2:
        return None

    hillas_pairs = list(itertools.combinations(hillas_parameters, 2))
    tel_config_pairs = list(itertools.combinations(tel_config, 2))
    print (tel_config_pairs)

    t1 = list(map(lambda t:[t[0]],[tel_config_pairs]))
    t1 = np.array(t1)
    t1 = np.transpose(t1)
    print (t1)

    h1 = list(map(lambda h:[h[0].psi,h[0].cen_x,h[0].cen_y,h[0].size],[hillas_pairs]))
    h1 = np.array(h1)
    h1 = np.transpose(h1)

    h2 = np.array(list(map(lambda h:[h[1].psi,h[1].cen_x,h[1].cen_y,h[1].size],hillas_pairs)))
    h2 = np.array(h2)
    h2 = np.transpose(h2)

    sx,sy = intersect_lines(h1[1],h1[2],h1[0],
                            h2[1],h2[2],h2[0])
    if weighting == "Konrad":
        weight_fn = weight_konrad
    elif weighting == "HESS":
        weight_fn = weight_HESS

    weight = weight_fn(h1[3],h2[3])
    weight *= weight_sin(h1[0],h2[0])

    sum_x = np.sum(sx*weight)
    sum_y = np.sum(sy*weight)
    sum_w = np.sum(weight)

    print (sum_x/sum_w,sum_y/sum_w)
    return sum_x/sum_w,sum_y/sum_w

def recontruct_tilted(hillas_parameters,weighting="Konrad"):

    return 0,0

def intersect_lines(xp1,yp1,phi1,xp2,yp2,phi2):

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
    return np.sin(np.fabs(phi1-phi2))