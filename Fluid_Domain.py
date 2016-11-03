import numpy as np
import math as math
import matplotlib.pyplot as plt
import sys
class Fluid_Domain:
    def __init__(self,fluid_input):
        bc_file = fluid_input + ".bc"
        self._read_boundary_file(bc_file)

        mesh = fluid_input + ".fgrid"


        try:
            fid = open(mesh, "r")
        except IOError:
            print("File '%s' not found." % mesh)
            sys.exit()
        ##########################################################################
        # NAMES TABLE
        # nverts: int,  number of fluid nodes
        # verts: float array[:,3], coordinates of fluid nodes
        # nelems: int, number of elements
        # elems: int array[:,3], element nodes id
        # nedges: int, number of edges
        # edges: int array[:,2], edge nodes id
        # connectivity: a list nverts lists, each list stores neighbor of corresponding verts
        # nbound_type: int, number of different boundary conditions
        # nbounds: int, number of boundary edges
        # bc_type: string list, name of different boundary conditions
        # bounds: int array[:,4], boundary start node id, boundary end node id, boundary type id
        # control_volume: double array[:], dual cell area
        # directed_area_vector: double array[nedges,2], directed area of each dual cell edge
        #                n2
        #    o-----------o-----------o
        #    |     dav   |  dav      |
        #    |       ^   |   ^       |
        #    |       |   |   |       |
        #    |   c - - - m - - -c    |
        #    |           |           |
        #    |           |           |    m: edge midpoint
        #    |           |           |    c: element centroid
        #    o-----------o-----------o
        #                n1
        #   
        #  shape_function_gradient = None;
        ####################################################################################################
        self.nverts = int(fid.readline())
        self.verts = np.empty(shape=[self.nverts,2],dtype=float)
        for i in xrange(self.nverts):
            self.verts[i,:] = map(float,fid.readline().split())
        self.nelems = int(fid.readline())
        self.elems = np.empty(shape=[self.nelems,3],dtype=int)
        for i in xrange(self.nelems):
            self.elems[i,:] = map(int,fid.readline().split())

        self.nedges = int(fid.readline())
        edge_info = np.empty(shape=[self.nedges,4],dtype=int)
        self.edges = np.empty(shape=[self.nedges,2],dtype=int)
        for i in xrange(self.nedges):
            edge_info[i,:] = map(int,fid.readline().split())
            self.edges[i,:] = edge_info[i,0:2]
        self.connectivity = [[] for i in xrange(self.nverts)]


        self.nbound_type,self.nbounds = map(int,fid.readline().split())

        self.bc_type = []
        self.bounds = np.empty(shape=[self.nbounds,4],dtype=int)
        bc_id = 0
        for i in xrange(self.nbound_type):
            n,bc = fid.readline().split()
            self.bc_type.append(bc)
            for j in xrange(int(n)):
                self.bounds[bc_id,0:3] = map(int,fid.readline().split())
                self.bounds[bc_id,3] = i
                bc_id += 1



        fid.close()

        self.control_volume = np.zeros(shape=[self.nverts,1],dtype=float);
        self.area = np.zeros(shape=[self.nelems],dtype=float);

        elem_center = np.zeros(shape=[self.nelems,2],dtype=float);
        for e in xrange(self.nelems):
            #####################################################
            # loop all the element compute dual cell area 
            #####################################################
            n1,n2,n3 = self.elems[e,:]

            x1,y1 = self.verts[n1,:]

            x2,y2 = self.verts[n2,:]

            x3,y3 = self.verts[n3,:]

            self.area[e] = e_area = 0.5*np.fabs((x2 - x1)*(y3 - y1) - (y2 - y1)*(x3 - x1))

            self.control_volume[n1,0] += e_area/3.0

            self.control_volume[n2,0] += e_area/3.0

            self.control_volume[n3,0] += e_area/3.0

            elem_center[e,:] = (x1+x2+x3)/3.0,(y1+y2+y3)/3.0


            
        self.directed_area_vector = np.zeros(shape=[self.nedges,2],dtype=float);





        for i in xrange(self.nedges):

            n1,n2,e1,e2 = edge_info[i,:]
            x1,y1 = self.verts[n1,:]
            x2,y2 = self.verts[n2,:]
            xm,ym = (x1+x2)/2.0, (y1+y2)/2.0

            if(e1 != -1):
                xc,yc = elem_center[e1,:]
                direction = np.sign((x1 - x2)*(yc - ym) - (xc - xm)*(y1 - y2))
                self.directed_area_vector[i] += direction*(ym - yc), direction*(xc - xm)

            if(e2 != -1):
                xc,yc = elem_center[e2,:]
                direction = np.sign((x1 - x2)*(yc - ym) - (xc - xm)*(y1 - y2))
                self.directed_area_vector[i] += direction*(ym - yc), direction*(xc - xm)


            self.connectivity[n1].append(n2)
            self.connectivity[n2].append(n1)





        self.shape_function_gradient = np.zeros(shape=[self.nelems,2,3],dtype=float);

        for e in xrange(self.nelems):

            ##################################################################
            #compute dphi_i /dx and dphi_i /dy, store in shape_shape_function
            #
            #shape_shape_function[e,:,:] = [ dphi_n1 /dx , dphi_n2 /dx  , dphi_n3 /dx
            #                                dphi_n1 /dy , dphi_n2 /dy  , dphi_n3 /dy
            #################################################################
            n1,n2,n3 = self.elems[e,:]

            x1,y1 = self.verts[n1,:]
            x2,y2 = self.verts[n2,:]
            x3,y3 = self.verts[n3,:]

            jac = 1.0/(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
            self.shape_function_gradient[e,0,:] = (y2-y3)*jac ,(y3-y1)*jac,(y1-y2)*jac
            self.shape_function_gradient[e,1,:] = (x3-x2)*jac ,(x1-x3)*jac,(x2-x1)*jac


        self.min_edge = np.empty(shape=[self.nverts,1],dtype=float)
        for n in xrange(self.nverts):

            ##################################################################
            #compute for each node i, min_{j} ||x_i - x_j||
            #################################################################
            self.min_edge[n,0] = np.inf
            x_n = self.verts[n,:]
            for m in self.connectivity[n]:
                x_m = self.verts[m,:]

                self.min_edge[n,0] = min(self.min_edge[n,0], np.linalg.norm(x_n-x_m))
        '''
        ###############
        # check edge_info
        ##############
        boundary_edge_number = 0
        for i in xrange(self.nedges):
            n,m,e1,e2 = edge_info[i,:]
            n1,n2,n3 = self.elems[e1,:]
            if((n != n1 and n !=n2 and n!= n3) or (m != n1 and m !=n2 and m!= n3)):
                print "error for edge %d, n is %d, m is %d, e1 is %d e2 is %d" %(i,n,m,e1,e2)

            if(e2 != -1):
                m1,m2,m3 = self.elems[e2,:]
                if((n != m1 and n !=m2 and n!= m3) or (m != m1 and m !=m2 and m!= m3)):
                    print "error for edge %d, n is %d, m is %d, e1 is %d e2 is %d" %(i,n,m,e1,e2)
            else:
                boundary_edge_number += 1
        if(boundary_edge_number != self.nbounds):
            print "error for boundary edge number"

        '''
    def _read_boundary_file(self,b_file):

        with open(b_file) as fid:
            for line in fid:
                lines = line.split()
                if(len(lines) < 2):
                    break

                if(lines[0] == "isothermal_wall"):
                     self.isothermal_wall = np.array(map(float, lines[1:]))
                elif(lines[0] == "adiabatic_wall"):
                     self.adiabatic_wall = np.array(map(float, lines[1:]))




    def check_mesh(self, draw = False):
        '''
        draw element
        '''
        #check the relation between edges and elements



        #check boundary edges elements relation, and boundary edges orientation
        for i in xrange(self.nbounds):
            n,m,e = self.bounds[i,0:3]
            n1,n2,n3 = self.elems[e,:]
            if((n != n1 and n !=n2 and n!= n3) or (m != n1 and m !=n2 and m!= n3)):
                print "error for boundary edge %d" %i
            l = n1 + n2 + n3 - n - m
            vec_12 = self.verts[n,:] - self.verts[l,:]
            vec_13 = self.verts[m,:] - self.verts[l,:]
            if(vec_12[0]*vec_13[1] - vec_12[1]*vec_13[0] <= 0.0):
                print "error for the orientation of boundary edge %d" %i

        #check area and control_volume
        all_area = 0
        for e in xrange(self.nelems):
            all_area += self.area[e]

        all_control_volume = 0
        for i in xrange(self.nverts):
            all_control_volume += self.control_volume[i,0]
        if( np.fabs(all_area - all_control_volume) > 1.0e-10):
            return "control volume error"


        if(draw):
            for i in xrange(self.nedges):
                n1,n2 = self.edges[i,:]
                x1,x2 = self.verts[n1],self.verts[n2]


                plt.plot([x1[0],x2[0]],[x1[1],x2[1]],color = 'r')
            for i in xrange(self.nbounds):

                n1,n2, e= self.bounds[i,0:3]
                x1,x2 = self.verts[n1],self.verts[n2]

                plt.plot([x1[0],x2[0]],[x1[1],x2[1]], color = 'b')

            for i in xrange(self.nedges):

                n1,n2 = self.edges[i,:]

                x1,x2 = self.verts[n1],self.verts[n2]

                xc = 0.5*(x1 + x2)

                xd = self.directed_area_vector[i]



                plt.plot([xc[0],xd[0] + xc[0]],[xc[1],xd[1] + xc[1]], color = 'g')

            plt.axis([-2, 2, -2, 2])
            plt.show()


        

'''
    def _lsq_gradient(self,V,status):


        for i in xrange(self.nverts):
            A =[]
            b = []
            xc = self.verts[i,:]
            Vc = self.V[i,:]
            for neighbor in self.connectivity[i]:
                if(status[neighbor]):#active node
                    xi = self.verts[neighbor]
                    Vi = self.verts[neighbor]
                    A.append(xi -xc)
                    b.append(Vi-Vc)


            A = np.array(A)
            b = np.array(b)

            self.gradientV[i,:,:] = np.linalg.lstsq(A,b)

    def _compute_stress_tensor(self,V,e,tau):

        elems = self.elems
        eos = self.eos

        n1,n2,n3 = elems[e,:]



        #########################
        # Compute velocity gradients, temperature gradients, store in d_v
        #####################


        vx1,vy1,T1 = V[n1,1],V[n1,2],eos._compute_temperature(V[n1,:])
        vx2,vy2,T2 = V[n1,1],V[n1,2],eos._compute_temperature(V[n2,:])
        vx3,vy3,T3 = V[n1,1],V[n1,2],eos._compute_temperature(V[n3,:])

        vx,vy,T = (vx1 + vx2 + vx3)/3.0,(vy1 + vy2 + vy3)/3.0,(T1 + T2 + T3)/3.0



        d_shape = self.shape_function_gradient[e,:,:]

        d_v = np.array([  [vx1*d_shape[0,0] + vx2*d_shape[0,1] + vx3*d_shape[0,2],vx1*d_shape[1,0] + vx2*d_shape[1,1] + vx3*d_shape[1,2]]
                          [vy1*d_shape[0,0] + vy2*d_shape[0,1] + vy3*d_shape[0,2],vy1*d_shape[1,0] + vy2*d_shape[1,1] + vy3*d_shape[1,2]]], dtype=float)

        #############################################
        # compute viscosity, heat conductivity
        ############################################


        vis_mu, vis_lambda, tem_k = eos._transport_coefficients(T)

        ############################
        # Compute stress tensor, store in tau
        ############################

        tau[0,0] = 2*vis_mu*d_v[0,0] + vis_lambda*(d_v[0,0] + d_v[1,1])
        tau[0,1] = vis_mu*(d_v[0,1] + d_v[1,0])
        tau[1,1] = 2*vis_mu*d_v[1,1] + vis_lambda*(d_v[0,0] + d_v[1,1])
        tau[1,0] = tau[0,1]



    def _compute_stress_tensor(self,V,e,q_flux):

          n1,n2,n3 = elems[e,:]



        #########################
        # Compute velocity gradients, temperature gradients, store in d_v
        #####################


        T1, T2, T3 = eos._compute_temperature(V[n1,:]) , eos._compute_temperature(V[n2,:]) , eos._compute_temperature(V[n3,:])


        T = (vx1 + vx2 + vx3)/3.0,(vy1 + vy2 + vy3)/3.0,(T1 + T2 + T3)/3.0



        d_shape = shape_function_gradient[e,:,:]

        d_v = np.array([  [vx1*d_shape[0,0] + vx2*d_shape[0,1] + vx3*d_shape[0,2],vx1*d_shape[1,0] + vx2*d_shape[1,1] + vx3*d_shape[1,2]]
                          [vy1*d_shape[0,0] + vy2*d_shape[0,1] + vy3*d_shape[0,2],vy1*d_shape[1,0] + vy2*d_shape[1,1] + vy3*d_shape[1,2]]
                          [ T1*d_shape[0,0] +  T2*d_shape[0,1] +  T3*d_shape[0,2], T1*d_shape[1,0] +  T2*d_shape[1,1] +  T3*d_shape[1,2]]], dtype=float)

        #############################################
        # compute viscosity, heat conductivity
        ############################################


        vis_mu, vis_lambda, tem_k = eos._transport_coefficients(T)




        #####################
        # Compute viscous flux
        ######################
        F_vis[:] =  0,tau[0,0],tau[0,1], tau[0,0]*vx + tau[0,1]*vy + tem_k*d_v[2,0]
        G_vis[:] =  0,tau[0,1],tau[1,1], tau[0,1]*vx + tau[1,1]*vy + tem_k*d_v[2,1]







'''

#fluid = Fluid_Domain("blasius_coarse.fgrid")
#fluid.check_mesh(True)
    