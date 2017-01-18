import numpy as np
class Limiter:
    def __init__(self,type,beta = 2.0):
        self.beta = beta
        self.type = type
    def _VA_slope_limiter(self,a,b,eps=1.0e-15):
        slope = np.empty(4,dtype=float)
        for i in range(4):
            if(a[i]*b[i]  < 0):
                slope[i] = 0.0
            else:
                slope[i] =  (a[i]*(b[i]**2 + eps) + b[i]*(a[i]**2 + eps))/(a[i]**2 + b[i]**2 + 2*eps)

        return slope
    #  vn *------* vm
    #  v_n v_m two primitive state variables
    #  dv_n  = \nabla v_n *(xm - xn)
    #  dv_m  = \nabla v_m *(xn - xm)

    def _reconstruct(self,v_n, v_m, dv_n, dv_m):
        if(self.type == 'None0'):
            return v_n, v_m
        if(self.type == 'None1'):
            return v_n + 0.5*dv_n, v_m + 0.5*dv_m
        if(self.type == 'Van_Albada'):
            v_L = v_n + 0.5*self._VA_slope_limiter(v_m - v_n, self.beta*dv_n + (1.0 - self.beta)*(v_m-v_n))
            v_R = v_m + 0.5*self._VA_slope_limiter(v_n - v_m, self.beta*dv_m + (1.0 - self.beta)*(v_n-v_m))
            return v_L, v_R
