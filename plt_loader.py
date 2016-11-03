import numpy as np
import sys
import matplotlib.pyplot as plt
mesh = "/home/icme-huang/Independence/2D_IBs_Euler-NS/TESTS/Blasius/blasius_00099.plt"



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
for i in xrange(n):
    lines = fid.readline().split()

    verts[i,:] = float(lines[0]) , float(lines[1])
    V[i,:] = float(lines[2]) , float(lines[3]), float(lines[4]) , float(lines[5])

for i in xrange(nelem):
    lines = fid.readline().split()
    elems[i,:] = int(lines[0]) , int(lines[1]), int(lines[2])
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
