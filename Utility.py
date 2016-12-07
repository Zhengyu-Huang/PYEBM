import numpy as np

def _interpolation(x_1, y_1, x_2, y_2, x_3):

    d2 = (x_2[0] - x_1[0])**2 + (x_2[1] - x_1[1])**2

    assert(d2 > 1e-10)

    alpha = np.sqrt(((x_3[0] - x_1[0])**2 + (x_3[1] - x_1[1])**2 )/ d2)

    l = len(y_1)

    return [y_1[i] + alpha*(y_2[i] - y_1[i]) for i in range(l)]