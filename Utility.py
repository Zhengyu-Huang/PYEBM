import numpy as np

def _interpolation(x_1, y_1, x_2, y_2, x_3):
    '''
    Interpolation value at x_3
    :param x_1: location of point1
    :param y_1: vector value at point1
    :param x_2: location of point2
    :param y_2: vector value at point2
    :param x_3: location of point3
    :return:
    '''


    d2 = (x_2[0] - x_1[0])**2 + (x_2[1] - x_1[1])**2

    assert(d2 > 1e-10)
    #alpha = (x3-x1,x2-x1)/(x2-x1,x2-x1)
    alpha = ((x_3[0] - x_1[0])*(x_2[0] - x_1[0]) + (x_3[1] - x_1[1])*(x_2[1] - x_1[1]))/ d2

    l = len(y_1)

    return [y_1[i] + alpha*(y_2[i] - y_1[i]) for i in range(l)]

def _interpolation2(y_1, y_2, alpha):
    '''
    Compute  y_1 + alpha*(y_2 - y_1)
    :param y_1: vector value at point1
    :param y_2: vector value at point2
    :param alpha: float
    :return:
    '''

    l = len(y_1)

    return [y_1[i] + alpha*(y_2[i] - y_1[i]) for i in range(l)]

def _interpolation_safe(x_1, y_1, x_2, y_2, x_3):
    '''
    Safe interpolation value at x_3, if it is extrapolation, y3 = y1 + min{alpha, 2}(y2 - y1)
                                                             y3 = y1 + max{alpha,-1}(y2 - y1)
    :param x_1: location of point1
    :param y_1: vector value at point1
    :param x_2: location of point2
    :param y_2: vector value at point2
    :param x_3: location of point3
    :return:
    '''

    d2 = (x_2[0] - x_1[0]) ** 2 + (x_2[1] - x_1[1]) ** 2

    assert (d2 > 1e-10)

    # alpha = (x3-x1,x2-x1)/(x2-x1,x2-x1)
    alpha = ((x_3[0] - x_1[0]) * (x_2[0] - x_1[0]) + (x_3[1] - x_1[1]) * (x_2[1] - x_1[1])) / d2
    if(alpha > 2.0):
        alpha = 2
    if(alpha < -1.0):
        alpha = -1.0

    l = len(y_1)

    return [y_1[i] + alpha * (y_2[i] - y_1[i]) for i in range(l)]