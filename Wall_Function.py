# This is a test file for different kind of nonlinear wall functions
import numpy as np
import matplotlib.pyplot as plt
class Wall_Function:
    def __init__(self):
        self.kappa = 0.41
        self.C_plus = 5.0

    def _Reichardt_function(self,y_plus):
        kappa = self.kappa
        return 2.5*np.log(1 + kappa*y_plus) + 7.8*(1 - np.exp(-y_plus/11) - y_plus/11*np.exp(-0.33*y_plus))

    def _log_layer(self,y_plus):
        kappa,C_plus = self.kappa, self.C_plus
        return 1/kappa*np.log(y_plus) + C_plus

    def _viscous_sublayer(self, y_plus):
        return y_plus

if __name__ == "__main__":
    y_plus = np.linspace(0,1000, 2000)
    wall_function = Wall_Function()

    u_plus_visc = wall_function._viscous_sublayer(y_plus)
    u_plus_log = wall_function._log_layer(y_plus)
    u_plus_log[0] = 0
    u_plus_Reichardt = wall_function._Reichardt_function(y_plus)
    plt.figure(1)
    plt.loglog(y_plus, u_plus_Reichardt,label="Reichardt")

    plt.loglog(y_plus, u_plus_visc,label="visc")

    plt.loglog(y_plus, u_plus_log,label="log")

    plt.legend()
    plt.show()
