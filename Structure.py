__author__ = 'zhengyuh'
import numpy as np
import sys
class Structure:
    def __init__(self,structure_input):
        ########################################################################
        ##### NAME TABLE
        ########################################################################
        # nverts: int, number of verts
        # verts: int array[nverts,2], coordinates of verts
        # nbounds: int, number of boundary element
        # bounds: int array[nbounds,2], boundary element verts id
        # vel: float array[nverts,2], velocity of boundary verts
        # edges_norm: float array[nbounds,2], edge outward norm, length is the edge length  
        # verts_norm: float array[nverts,2], vert outward norm, average of two neighbor edge norms


        mesh = structure_input + ".sgrid"

        try:
            fid = open(mesh, "r")
        except IOError:
            print("File '%s' not found." % mesh)
            sys.exit()
        self.nverts = int(fid.readline())
        self.verts = np.empty(shape=[self.nverts,2],dtype=float)
        for i in range(self.nverts):
            self.verts[i,:] = [float(x) for x in fid.readline().split()]
        self.nbounds = int(fid.readline())
        self.bounds = np.empty(shape=[self.nbounds,2],dtype=int)
        for i in range(self.nbounds):
            self.bounds[i,:] = [int(x) for x in fid.readline().split()]

        self.vel = np.zeros(shape=[self.nverts,2],dtype=float)
        self.edges_norm = np.zeros(shape=[self.nbounds,2],dtype=float)
        self.verts_norm = np.zeros(shape=[self.nverts,2],dtype=float)

        fid.close()
        self._construct_norm()




    def _construct_norm(self):
        '''
        The norm direction is toward the center of the structure, because the
        structure edge is clockwise
        :return:
        '''
        bounds = self.bounds
        verts = self.verts
        edges_norm = self.edges_norm
        verts_norm = self.verts_norm

        for i in range(self.nbounds):
            n1,n2 = bounds[i,:]
            x1,x2 = verts[n1,:],verts[n2,:]
            edges_norm[i,:] = [x2[1] - x1[1], x1[0] - x2[0]]
            verts_norm[n1,:] += 0.5*edges_norm[i,:]
            verts_norm[n2,:] += 0.5*edges_norm[i,:]


    def _point_info(self,i,alpha, x):
        #input:
        #     edge id :i
        #     local coordinate of point in the segment: x_p = x1 + alpha *(x2 - x1)
        #     x: a point to determine the normal direction
        #return:
        #     point coordinate : x_p
        #     structure norm at x_p : norm_p
        #     structure velocity at x_p : v_p
        verts = self.verts
        bounds = self.bounds
        vel = self.vel
        verts_norm = self.verts_norm
        edges_norm = self.edges_norm

        n1,n2 = bounds[i,:]
        x1,x2 = verts[n1,:],verts[n2,:]
        v1,v2 = vel[n1,:],vel[n2,:]

        x_p = (1 - alpha)*x1 + alpha*x2
        v_p = (1 - alpha)*v1 + alpha*v2

        if(alpha < 1e-16): # the point is the left node point
            norm_p = verts_norm[n1,:]
        elif(alpha > 1 - 1e-16): # the point is the right node point
            norm_p = verts_norm[n2,:]
        else:# the point on the edge
            norm_p = edges_norm[i,:]

        if (x[0]-x_p[0])*norm_p[0] + (x[1]-x_p[1])*norm_p[1] < 0:
            norm_p = - norm_p

        return x_p, v_p, norm_p

    def _temperature(self,i, alpha):
        return 1



#struc = Structure('plate.fgrid')
