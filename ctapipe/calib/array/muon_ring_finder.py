import numpy as np


def chaudhuri_kundu_circle_fit(x,y,weight):
    """
    Fast and reliable analytical circle fitting method previously used in the H.E.S.S.
    experiment for muon identification

    Implementation based on:
    Chaudhuri/Kundu "Optimum circular fit to weighted data in multi-dimensional space"
    Pattern Recognition Letters 14 (1993) pp.1-6

    :param x: X position of pixel [ndarray]
    :param y: Y position of pixel [ndarray]
    :param weight: weighting of pixel in fit [ndarray]
    :return: X position, Y position and radius of circle
    """
    #First calculate the weighted average positions of the pixels
    sum_weight = np.sum(weight)
    av_weighted_pos_x = np.sum(x * weight)/sum_weight
    av_weighted_pos_y = np.sum(y * weight)/sum_weight

    #The following notation is a bit ugly but directly references the paper notation
    factor = np.power(x,2)+np.power(y,2)

    a = np.sum(weight * (x - av_weighted_pos_x) * x)
    a_prime = np.sum(weight * (y - av_weighted_pos_y) * x)

    b = np.sum(weight * (x - av_weighted_pos_x) * y)
    b_prime = np.sum(weight * (y - av_weighted_pos_y) * y)

    c = np.sum(weight * (x - av_weighted_pos_x) * factor) * 0.5
    c_prime = np.sum(weight * (y - av_weighted_pos_y) * factor) * 0.5

    nom_0 = ((a* b_prime) - (a_prime * b))
    nom_1 = ((a_prime * b) - (a * b_prime))

    #Calculate circle centre and radius
    centre_x = ((b_prime * c) - (b * c_prime)) / nom_0
    centre_y = ((a_prime * c) - (a * c_prime)) / nom_1

    radius = np.sqrt( np.sum(weight * (np.power(x - centre_x,2) + np.power(y - centre_y,2)))/sum_weight )
    return centre_x,centre_y,radius

