import numpy as np
import sys
import matplotlib.pyplot as plt
from EOS import *

def _pri_to_conser_all(V, W):
    gamma = 1.4
    W[:, 0] = V[:, 0]
    W[:, 1] = V[:, 1] * V[:, 0]
    W[:, 2] = V[:, 2] * V[:, 0]
    W[:, 3] = 0.5 * V[:, 0] * (V[:, 1] ** 2 + V[:, 2] ** 2) + V[:, 3] / (gamma - 1.0)


def _conser_to_pri_all(W, V):
    gamma = 1.4
    V[:, 0] = W[:, 0]
    V[:, 1] = W[:, 1] / W[:, 0]
    V[:, 2] = W[:, 2] / W[:, 0]
    V[:, 3] = (W[:, 3] - 0.5 * W[:, 1] * V[:, 1] - 0.5 * W[:, 2] * V[:, 2]) * (gamma - 1.0)


mesh = "/home/zhengyuh/Independence/2D_IBs_Euler-NS/TESTS/EulerNaca/domain_00099.dat"



try:
    fid = open(mesh, "r")
except IOError:
    print("File '%s' not found." % mesh)
    sys.exit()
line = fid.readline() # variable name xy rho u v p M id
line = fid.readline().split(',')
n = int(line[1].split('=')[1])
nelem = int(line[2].split('=')[1])
verts = np.empty(shape = (n,2), dtype = float)
V = np.empty(shape = (n,4), dtype = float)
elems = np.empty(shape = (nelem,3), dtype = float)
for i in range(n):
    lines = fid.readline().split()

    verts[i,:] = float(lines[0]) , float(lines[1])
    V[i,:] = float(lines[2]) , float(lines[3]), float(lines[4]) , float(lines[5])

for i in range(nelem):
    lines = fid.readline().split()
    elems[i,:] = int(lines[0]) , int(lines[1]), int(lines[2])

fid.close()

W = np.empty(shape= (n,4), dtype = float)
_pri_to_conser_all(V,W)
np.save("ns_IB_dante_lim1",W)


W_me = np.load("nacaW.npy")
gamma = 1.4
V_me = np.empty(shape= (n,4), dtype = float)
_conser_to_pri_all(W_me,V_me)
V = V - V_me


elems -=1
x , y  = verts[:,0],verts[:,1]
rho,vx,vy,p = V[:,0],V[:,1],V[:,2],V[:,3]
plt.figure(1)
plt.tripcolor(x, y, elems, rho,shading='gouraud', cmap=plt.cm.rainbow)
plt.title('density')
plt.colorbar()
# set the limits of the plot to the limits of the data
plt.axis([x.min(), x.max(), y.min(), y.max()])

plt.figure(2)
plt.tripcolor(x, y, elems, p,shading='gouraud', cmap=plt.cm.rainbow)
plt.title('pressure')
plt.colorbar()
# set the limits of the plot to the limits of the data
plt.axis([x.min(), x.max(), y.min(), y.max()])


plt.figure(4)
plt.tripcolor(x, y, elems, vx , shading='gouraud', cmap=plt.cm.rainbow)
plt.title('Vx')
plt.colorbar()
# set the limits of the plot to the limits of the data
plt.axis([x.min(), x.max(), y.min(), y.max()])

plt.figure(5)
plt.tripcolor(x, y, elems, vy , shading='gouraud', cmap=plt.cm.rainbow)
plt.title('Vy')
plt.colorbar()
# set the limits of the plot to the limits of the data
plt.axis([x.min(), x.max(), y.min(), y.max()])

plt.show()
