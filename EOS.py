import numpy as np

class EOS:
    def __init__(self, gamma,  Prandtl = 0.72 , Mach = 1.0, Reynolds = 100):
        self.gamma = gamma
        self.Prandtl = Prandtl
        self.mu = Mach/Reynolds
    def _transport_coefficients(self):

        #return mu mu_lambda and kappa

        return self.mu , -2.0*self.mu/3.0, self.mu/(self.Prandtl *(self.gamma - 1))


    #change primitive variables with conservative variables
    def _pri_to_conser(self, V):
        gamma = self.gamma
        [rho, vx, vy, p] = V;
        return np.array([rho,rho*vx, rho*vy,rho*(vx*vx + vy*vy)/2 + p/(gamma - 1)],dtype=float)
    def _compute_temperature(self, V):
         return self.gamma*V[3]/V[0]

    def _conser_to_pri(self, W):
        gamma = self.gamma
        [w1, w2, w3, w4] = W
        rho = w1;
        vx = w2/w1;
        vy = w3/w1
        p = (w4 - w2*vx/2 - w3*vy/2) * (gamma - 1)

        return np.array([rho,vx,vy,p],dtype=float)



    def _pri_to_conser_all(self,V,W):
        gamma = self.gamma
        W[:,0] = V[:,0]
        W[:,1] = V[:,1]*V[:,0]
        W[:,2] = V[:,2]*V[:,0]
        W[:,3] = 0.5*V[:,0]*(V[:,1]**2 + V[:,2]**2) + V[:,3]/(gamma - 1.0)

    #change conservative variables to primitive variables
    def _conser_to_pri_all(self,W, V):
        gamma = self.gamma
        V[:,0] = W[:,0]
        V[:,1] = W[:,1]/W[:,0]
        V[:,2] = W[:,2]/W[:,0]
        V[:,3] = (W[:,3] - 0.5*W[:,1]*V[:,1] - 0.5*W[:,2]*V[:,2]) * (gamma - 1.0)


    def _conser_to_pri_temp(self, w,eos):

        R,gamma = eos.R, eos.gamma

        [w1, w2, w3, w4] = w
        rho = w1;
        vx = w2/w1;
        vy = w3/w1
        p = (w4 - w2*vx/2 - w3*vy/2) * (gamma - 1)
        T = p/(rho*R)

        return np.array([rho, vx, vy, T])