import numpy as np


def kundu_chaudhuri_circle_fit(x, y, weights):
    '''
    Fast, analytic calculation of circle center and radius from
    x, y and weights. This should be pixel positions and photon equivalents

    Reference:
    B. B. Chaudhuri und P. Kundu.
    "Optimum circular fit to weighted data in multi-dimensional space".
    In: Pattern Recognition Letters 14.1 (1993), S. 1–6
    '''


    mean_x = np.average(x, weights=weights)
    mean_y = np.average(y, weights=weights)

    a1 = np.sum(weights * (x - mean_x) * x)
    a2 = np.sum(weights * (y - mean_y) * x)

    b1 = np.sum(weights * (x - mean_x) * y)
    b2 = np.sum(weights * (y - mean_y) * y)

    c1 = 0.5 * np.sum(weights * (x - mean_x) * (x**2 + y**2))
    c2 = 0.5 * np.sum(weights * (y - mean_y) * (x**2 + y**2))

    center_x = (b2 * c1 - b1 * c2) / (a1 * b2 - a2 * b1)
    center_y = (a2 * c1 - a1 * c2) / (a2 * b1 - a1 * b2)

    radius = np.sqrt(np.average(
        (center_x - x)**2 + (center_y - y)**2,
        weights=weights,
    ))

    return radius, center_x, center_y
