from Limiter import *
from Intersector import *
from EOS import EOS
import Flux
from Fluid_Domain import *


class Explicit_Solver:
    #gradientV: double array[:,2,4], gradient of primitive variables at each nodes
    def __init__(self,fluid_domain,io_data):

        self.fluid_domain = fluid_domain

        nverts = fluid_domain.nverts

        self.W = np.empty(shape=[nverts,4],dtype=float)

        self.k1 = np.empty(shape=[nverts,4],dtype=float)

        self.k2 = np.empty(shape=[nverts,4],dtype=float)

        #####################################################
        # construct eos
        ####################################################
        self.eos = EOS(io_data.gamma,io_data.Prandtl,io_data.Mach,io_data.Reynolds)

        #####################################################
        # construct limiter
        ####################################################
        self.limiter = Limiter(io_data.limiter)

        ####################################################
        # equation type, Euler or NS
        ####################################################
        self.equation_type = io_data.equation_type

        ####################################################
        # CFL number
        ####################################################
        self.CFL = 0.3

        #####################################################
        # Tolerance and residual
        ####################################################
        self.tol = io_data.tolerance
        self.res_L2 = 0.0
        self.res_L2_ref = np.inf
        self.max_ite = io_data.max_ite

        #####################################################
        # construct non-dimensional initial state
        #####################################################
        # rho_oo = 1
        # p_oo = 1/gamma
        # u_oo = M
        #####################################################

        V_oo = np.array([1.0 , io_data.Mach*np.cos(io_data.AoA*np.pi/180.0), io_data.Mach*np.sin(io_data.AoA*np.pi/180.0) , 1/self.eos.gamma],dtype=float)

        self.W_oo = self.eos._pri_to_conser(V_oo)

        self.gradient_V = np.empty(shape=[fluid_domain.nverts,2,4], dtype = float)






    def _init_fluid_state(self):

        fluid = self.fluid_domain


        W = self.W

        nverts = fluid.nverts

        for i in xrange(nverts):

            W[i,:] = self.W_oo

        self._apply_wall_boundary_condition(W)
        self.W = np.load("solutionW.npy")



    def _solve(self):

        self._init_fluid_state()

        ite = 0

        while ite < self.max_ite:

            self._steady_time_advance()

            ite += 1

            print "ite %d ,relative residual is %.10f" %(ite,self.res_L2)

            if(self.res_L2 <= self.tol):
                break



    def _steady_time_advance(self):

        fluid = self.fluid_domain


        W, k1, k2 = self.W,  self.k1, self.k2

        eos = self.eos

        V=np.empty(shape=[fluid.nverts,4],dtype=float)

        eos._conser_to_pri_all(W,V)

        dt = self._compute_local_time_step(V)

        control_volume = fluid.control_volume
        #print "at begin ",30, V[30,:]


        k1[:,:] = 0

        self._compute_RK_update(V, k1);



        W0 = W + k1*dt/control_volume;



        self._apply_wall_boundary_condition(W0);

        self._check_solution(W0);

        k2[:,:] = 0

        eos._conser_to_pri_all(W0,V)

        self._compute_RK_update(V, k2);

        R = 1.0/2.0 * (k1 + k2)



        W +=  R*dt/control_volume;

        self._apply_wall_boundary_condition(W);

        self._check_solution(W);

        self._compute_residual(R,W)









    def _compute_RK_update(self, V, R):
        #compute euler flux



        self._euler_flux_rhs(V, R)
        #print "euler R", R[30,:]

        self._euler_boundary_flux(V, R)
        #print "euler boundary R", R[30,:]

        #compute diffusion term
        if(self.equation_type == "NS"):

            self._viscid_flux_rhs_fem(V,R)

            self._viscid_boundary_flux_fem(V, R)

    def _lsq_gradient(self,V):

        fluid = self.fluid_domain
        for i in xrange(fluid.nverts):
            A =[]
            b = []
            xc = fluid.verts[i,:]
            Vc = V[i,:]
            for neighbor in fluid.connectivity[i]:

                xi = fluid.verts[neighbor]
                Vi = V[neighbor]
                A.append(xi - xc)
                b.append(Vi - Vc)

            A = np.array(A)
            b = np.array(b)

            self.gradient_V[i,:,:] = np.linalg.lstsq(A,b)[0]

    def _euler_flux_rhs(self,V,R):
        # convert conservative state variable W to primitive state variable V

        fluid = self.fluid_domain

        limiter = self.limiter
        eos = self.eos

        self._lsq_gradient(V)

        for i in xrange(fluid.nedges):

            n,m = fluid.edges[i,:]

            v_n,v_m = V[n,:],V[m,:]

            dr_nm = fluid.directed_area_vector[i,:]

            dv_n,dv_m = np.dot(dr_nm,self.gradient_V[n,:,:]),np.dot(dr_nm,self.gradient_V[m,:,:])

            v_L, v_R =limiter._reconstruct(v_n, v_m, dv_n, dv_m)

            flux = Flux._Roe_flux(v_L,v_R,dr_nm,eos)


            R[n,:] -= flux
            R[m,:] += flux



        #R*dt/vol

    def _euler_boundary_flux(self,V,R):

        fluid = self.fluid_domain
        eos = self.eos
        for i in xrange(fluid.nbounds):

            m,n,e,type = fluid.bounds[i,:]

            x_m = fluid.verts[m,:]

            x_n = fluid.verts[n,:]

            prim_m = V[m,:]

            prim_n = V[n,:]

            dr_nm = 0.5*np.array([x_n[1] - x_m[1], x_m[0] - x_n[0]],dtype=float)

            if(fluid.bc_type[type] == "free_stream"):

                W_oo = self.W_oo

                R[m,:] -= Flux._Steger_Warming(prim_m, W_oo, dr_nm, eos)

                R[n,:] -= Flux._Steger_Warming(prim_n, W_oo, dr_nm, eos)

            elif(fluid.bc_type[type] == "slip_wall" or fluid.bc_type[type] == "adiabatic_wall" or fluid.bc_type[type] == "isothermal_wall" ):

                R[m,:] -= [0.0, prim_m[3]*dr_nm[0] , prim_m[3]*dr_nm[1], 0.0] #Flux._exact_flux(prim_m, dr_nm, eos)

                R[n,:] -= [0.0, prim_n[3]*dr_nm[0] , prim_n[3]*dr_nm[1], 0.0] #Flux._exact_flux(prim_n, dr_nm, eos)

    def _viscid_flux_rhs_fem(self,V,R):
        eos = self.eos
        nelems = self.fluid_domain.nelems
        elems = self.fluid_domain.elems
        shape_function_gradient = self.fluid_domain.shape_function_gradient
        area = self.fluid_domain.area

        tau = np.empty([2,2],dtype=float)
        F_vis = np.empty(4,dtype=float)
        G_vis = np.empty(4,dtype=float)
        for e in range(nelems):

            n1,n2,n3 = elems[e,:]



            #########################
            # Compute velocity gradients, temperature gradients, store in d_v
            #####################


            vx1,vy1,T1 = V[n1,1],V[n1,2],eos._compute_temperature(V[n1,:])
            vx2,vy2,T2 = V[n2,1],V[n2,2],eos._compute_temperature(V[n2,:])
            vx3,vy3,T3 = V[n3,1],V[n3,2],eos._compute_temperature(V[n3,:])

            vx,vy,T = (vx1 + vx2 + vx3)/3.0,(vy1 + vy2 + vy3)/3.0,(T1 + T2 + T3)/3.0



            d_shape = shape_function_gradient[e,:,:]

            d_v = np.array([  [vx1*d_shape[0,0] + vx2*d_shape[0,1] + vx3*d_shape[0,2],vx1*d_shape[1,0] + vx2*d_shape[1,1] + vx3*d_shape[1,2]],
                              [vy1*d_shape[0,0] + vy2*d_shape[0,1] + vy3*d_shape[0,2],vy1*d_shape[1,0] + vy2*d_shape[1,1] + vy3*d_shape[1,2]],
                              [ T1*d_shape[0,0] +  T2*d_shape[0,1] +  T3*d_shape[0,2], T1*d_shape[1,0] +  T2*d_shape[1,1] +  T3*d_shape[1,2]]], dtype=float)

            #############################################
            # compute viscosity, heat conductivity
            ############################################


            vis_mu, vis_lambda, tem_k = eos._transport_coefficients()

            ############################
            # Compute stress tensor, store in tau
            ############################

            tau[0,0] = 2*vis_mu*d_v[0,0] + vis_lambda*(d_v[0,0] + d_v[1,1])
            tau[0,1] = vis_mu*(d_v[0,1] + d_v[1,0])
            tau[1,1] = 2*vis_mu*d_v[1,1] + vis_lambda*(d_v[0,0] + d_v[1,1])
            tau[1,0] = tau[0,1]


            #####################
            # Compute viscous flux
            ######################
            F_vis[:] =  0,-tau[0,0],-tau[0,1], -tau[0,0]*vx - tau[0,1]*vy - tem_k*d_v[2,0]
            G_vis[:] =  0,-tau[0,1],-tau[1,1], -tau[0,1]*vx - tau[1,1]*vy - tem_k*d_v[2,1]



            e_area = area[e]
            R[n1,:] += e_area*d_shape[0,0]*F_vis + e_area*d_shape[1,0]*G_vis
            R[n2,:] += e_area*d_shape[0,1]*F_vis + e_area*d_shape[1,1]*G_vis
            R[n3,:] += e_area*d_shape[0,2]*F_vis + e_area*d_shape[1,2]*G_vis


    def _viscid_boundary_flux_fem(self, V, R):

        fluid = self.fluid_domain

        eos = self.eos

        tau = np.empty([2,2],dtype=float)

        vis_boundary = np.zeros(4,dtype=float)



        for i in xrange(fluid.nbounds):

            n,m,e,type = fluid.bounds[i,:]

            x_n = fluid.verts[n,:]

            x_m = fluid.verts[m,:]

            nx,ny = 0.5*np.array([x_n[1] - x_m[1], x_m[0] - x_n[0]],dtype=float)

            if(fluid.bc_type[type] == "adiabatic_wall"):
                vx_wall, vy_wall, q_flux = fluid.adiabatic_wall

                n1,n2,n3 = fluid.elems[e,:]

                vx1,vy1,T1 = V[n1,1],V[n1,2],eos._compute_temperature(V[n1,:])

                vx2,vy2,T2 = V[n1,1],V[n1,2],eos._compute_temperature(V[n2,:])

                vx3,vy3,T3 = V[n1,1],V[n1,2],eos._compute_temperature(V[n3,:])



                d_shape = self.shape_function_gradient[e,:,:]

                d_v = np.array([  [vx1*d_shape[0,0] + vx2*d_shape[0,1] + vx3*d_shape[0,2],vx1*d_shape[1,0] + vx2*d_shape[1,1] + vx3*d_shape[1,2]]
                                  [vy1*d_shape[0,0] + vy2*d_shape[0,1] + vy3*d_shape[0,2],vy1*d_shape[1,0] + vy2*d_shape[1,1] + vy3*d_shape[1,2]]
                                  [ T1*d_shape[0,0] +  T2*d_shape[0,1] +  T3*d_shape[0,2], T1*d_shape[1,0] +  T2*d_shape[1,1] +  T3*d_shape[1,2]]], dtype=float)

                #############################################
                # compute viscosity, heat conductivity
                ############################################


                vis_mu, vis_lambda, tem_k = eos._transport_coefficients( )

                ############################
                # Compute stress tensor, store in tau
                ############################

                tau[0,0] = 2*vis_mu*d_v[0,0] + vis_lambda*(d_v[0,0] + d_v[1,1])
                tau[0,1] = vis_mu*(d_v[0,1] + d_v[1,0])
                tau[1,1] = 2*vis_mu*d_v[1,1] + vis_lambda*(d_v[0,0] + d_v[1,1])
                tau[1,0] = tau[0,1]


                #####################
                # Compute viscous flux
                ######################
                vis_boundary[3] = - (tau[0,0]*vx_wall + tau[0,1]*vy_wall)*nx - (tau[0,1]*vx_wall + tau[1,1]*vy_wall)*ny

                R[n,:] -= vis_boundary
                R[m,:] -= vis_boundary

    def _apply_wall_boundary_condition(self,W):

        fluid = self.fluid_domain

        eos = self.eos

        for i in xrange(fluid.nbounds):

            n,m,e,type = fluid.bounds[i,:]

            if(fluid.bc_type[type] == "slip_wall"):
                continue


            if(fluid.bc_type[type] == "adiabatic_wall"):
                W[m,3] += 0.5*W[n,0]*(fluid.adiabatic_wall[0]**2 + fluid.adiabatic_wall[1]**2 - W[m,1]**2 - W[m,2]**2)
                W[n,3] += 0.5*W[n,0]*(fluid.adiabatic_wall[0]**2 + fluid.adiabatic_wall[1]**2 - W[n,1]**2 - W[n,2]**2)
                W[n,1:3] = W[n,0]*fluid.adiabatic_wall[0:2]
                W[m,1:3] = W[m,0]*fluid.adiabatic_wall[0:2]

            elif(fluid.bc_type[type] == "isothermal_wall"):



                W[n,1:3] = W[n,0]*fluid.isothermal_wall[0:2]
                W[m,1:3] = W[m,0]*fluid.isothermal_wall[0:2]

                e_wall = fluid.isothermal_wall[2] /(eos.gamma*(eos.gamma-1)) + 0.5*fluid.isothermal_wall[0]**2 + 0.5*fluid.isothermal_wall[1]**2
                W[n,3] = W[n,0]*e_wall
                W[m,3] = W[m,0]*e_wall

    def _compute_local_time_step(self,V):

        c = np.sqrt(self.eos.gamma * V[:,3]/ V[:,0])
        wave_speed = np.sqrt(V[:,1]**2 + V[:,2]**2) + c # + self.eos.mu

        dt = self.CFL*self.fluid_domain.min_edge/wave_speed[:,None]

        return dt

    def _compute_residual(self,R,W):

        fluid = self.fluid_domain

        eos = self.eos

        for i in xrange(fluid.nbounds):

            n,m,e,type = fluid.bounds[i,:]
            #change the flux to the difference between the real wall state and the computed one


            if(fluid.bc_type[type] == "adiabatic_wall" or fluid.bc_type[type] == "isothermal_wall"):
                R[m,1] = W[m,1] - W[m,0]*fluid.adiabatic_wall[0]
                R[m,2] = W[m,2] - W[m,0]*fluid.adiabatic_wall[1]

                R[n,1] = W[n,1] - W[n,0]*fluid.adiabatic_wall[0]
                R[n,2] = W[n,2] - W[n,0]*fluid.adiabatic_wall[1]



            elif(fluid.bc_type[type] == "isothermal_wall"):


                e_wall = fluid.isothermal_wall[2] /(eos.gamma*(eos.gamma-1)) + 0.5*fluid.isothermal_wall[0]**2 + 0.5*fluid.isothermal_wall[1]**2
                R[m,3] = W[m,3] - W[m,0]*e_wall
                R[n,3] = W[n,3] - W[n,0]*e_wall

        self.res_L2 = np.linalg.norm(R)

        if(np.isinf(self.res_L2_ref)):

            self.res_L2_ref = self.res_L2 *self.tol


    def _check_solution(self,W):

        V = np.empty(shape=[self.fluid_domain.nverts,4],dtype=float)

        self.eos._conser_to_pri_all(W,V)

        rho,p = V[:,0], V[:,3]

        if ((rho > 0).all()):
            return;
        else:
            print "negative density"

        if ((p > 0).all()):
            return;
        else:
            print "negative density"


    def _draw_solution(self):

        W = self.W

        fluid = self.fluid_domain

        eos = self.eos

        V = np.empty(shape=[fluid.nverts,4],dtype=float)

        self.eos._conser_to_pri_all(W,V)

        Mach = np.sqrt((V[:,1]**2 + V[:,2]**2)*V[:,0]/(eos.gamma*V[:,3]))

        x = fluid.verts[:,0]

        y = fluid.verts[:,1]

        elems = fluid.elems

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

        plt.figure(3)
        plt.tripcolor(x, y, elems, Mach, shading='gouraud', cmap=plt.cm.rainbow)
        plt.title('Mach number')
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
        return



'''
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
class Embedded_Explicit_Solver:
    def __init__(self,fluid_domain,intersector,io_data):

        self.fluid_domain = fluid_domain

        self.intersector = intersector

        nverts = fluid.nverts

        self.W = np.empty(shape=[nverts,4],dtype=float)

        self.k1 = np.empty(shape=[nverts,4],dtype=float)

        self.k2 = np.empty(shape=[nverts,4],dtype=float)

        #####################################################
        # construct eos
        ####################################################
        self.eos = eos = EOS(io_data.gamma,io_data.Prandtl)

        #####################################################
        # construct limiter
        ####################################################
        self.limiter = Limiter(io_data.limiter)

        ####################################################
        # equation type, Euler or NS
        ####################################################
        self.equation_type = io_data.equation_type

        #####################################################
        # construct non-dimensional initial state
        #####################################################
        # rho_oo = 1
        # p_oo = 1/gamma
        # u_oo = M
        #####################################################

        V_oo = np.array([1.0 , io_data.Mach*np.cos(io_data.AoA), io_data.Mach*np.sin(io_data.AoA) , 1/eos.gamma],dtype=float)

        self.W_oo = eos._pri_to_conser(V_oo)
    
        nverts = self.fluid_domain.nverts

        W = self.W

        for i in xrange(nverts):
            W[i,:] = self.W_oo
        for i in xrange(fluid.nbounds):

            m,n,type = fluid.bounds[i,:]

            x_n = self.fluid_domain.verts[n,:]
            x_m = self.fluid_domain.verts[m,:]

            dr_nm = np.array([x_n[1] - x_m[1], x_m[0] - x_n[0]],dtype=float)
            dr_nm = dr_nm/np.linalg.norm(dr_nm)

            if(fluid.bc_type[type] == "slip_wall"):

                v_m = eos._conser_to_prim(W[m,:])

                v_m[1:3] -= np.dot(v_m[1:3]*dr_nm) * dr_nm

                W[m,:] = eos._prim_to_conser(v_m)

                v_n = eos._conser_to_prim(W[n,:])

                v_n[1:3] -= np.dot(v_n[1:3]*dr_nm) * dr_nm

                W[n,:] = eos._prim_to_conser(v_n)





    def _steady_time_advance(self):


        W, k1, k2 = self.W,  self.k1, self.k2

        eos = self.eos

        V=np.empty(shape=[fluid.nverts,4],dtype=float)

        eos._conser_to_pri_all(W,V)

        dt = self._compute_time_step(V,eos)

        control_volume = self.fluid_domain.control_volume

        k1[:,:] = 0

        self._compute_RK_update(W, k1);



        W0 = W + k1*dt/control_volume;

        self._apply_wall_boundary_condition(W0);

        self._check_solution(W0);

        k2[:,:] = 0

        self._compute_RK_update(W0, k2);

        W +=  1.0/2.0 * (k1 + k2)*dt/control_volume;


        self._apply_wall_boundary_condition(W);

        self.check_solution(W);










    def _compute_RK_update(self, V, R):
        #compute euler flux

        self._euler_flux_rhs(V, R)

        self._euler_boundary_flux(V, R)









        #self._viscid_flux_rhs_fem(t, with_dt)

    def _euler_flux_rhs(self,V,R):
        # convert conservative state variable W to primitive state variable V

        fluid = self.fluid_domain
        status = self.intersector.status
        intersect_or_not = self.intersector.intersect_or_not
        limiter = self.limiter
        eos = self.eos

        fluid._lsq_gradient(status)




        for i in xrange(fluid.nedges):
            n,m = fluid.edges[i,:]
            v_n,v_m = V[n,:],V[m,:]
            n_active,m_active = status[n],status[m]
            intersect = intersect_or_not[i]
            dr_nm = fluid.edge_norm[i,:]
            dv_n,dv_m = np.dot(dr_nm,fluid.gradient_V[n,:,:]),np.dot(dr_nm,fluid.gradient_V[m,:,:])
            if(n_active and m_active and not intersect):

                v_L, v_R =limiter._reconstruct(v_n, v_m, dv_n, dv_m)

            elif(n_active and not m_active):
                v_L = v_n + 0.5*dv_n
                v_R = self.SIV(i,self.intersector)
            elif(m_active and not n_active):
                v_R = v_m + 0.5*dv_m
                v_L = self.SIV(i,self.intersector)
            flux = _Roe_flux(v_L,v_R,dr_nm,eos)

            R[n,:] -= flux
            R[m,:] += flux

        #R*dt/vol



    def _euler_boundary_flux(self,V,R):

        fluid = self.fluid_domain
        eos = self.eos
        for i in xrange(fluid.nbounds):

            m,n,type = fluid.bounds[i,:]

            x_m = fluid.verts[m,:]

            x_n = fluid.verts[n,:]

            prim_m = V[m,:]

            prim_n = V[n,:]

            dr_nm = 0.5*np.array([x_n[1] - x_m[1], x_m[0] - x_n[0]],dtype=float)

            if(fluid.bc_type[type] == "free_stream"):

                W_oo = fluid.free_stream

                R[m,:] -= _Steger_Warming(prim_m, W_oo, dr_nm, eos)

                R[n,:] -= _Steger_Warming(prim_n, W_oo, dr_nm, eos)

            elif(fluid.bc_type[type] == "slip_wall"):

                R[m,:] -= _exact_flux(prim_m, dr_nm, eos)

                R[n,:] -= _exact_flux(prim_n, dr_nm, eos)







    def _fem_ghost_node_update(self):
        return ;

    def _viscid_flux_rhs_fem(self):
        #for e in xrange()
        #    check element e valid
        return ;





    def SIV(self,V, i,limiter):
        #p---c----q
        #    \
        #     \
        #  n1  si  n2
        #      \
        #       \
        #        b

        verts = self.fluid.verts
        gradient_V = self.fluid.gradient_V

        n1,n2 = self.fluid.edges[i,:]
        eos = self.eos
        status = intersector.status
        x_si = 0.5*(verts[n1,:] + verts[n2,:])

        n_p, n_q, alpha_c = self.intersector.edge_center_stencil[i]

        #Information of closest point on the structure
        struc_i,struc_alpha = self.intersector.edge_center_closest_position[i]
        x_b,vv_wall,nn_wall = self.structure._point_info(struc_i,struc_alpha)

        #Construct x_c information, including primitive variable v_c, dv_c
        x_p,x_q = verts[n_p,:],verts[n_q,:]
        x_c = alpha_c*x_p + (1 - alpha_c)*x_q
        if(status[n_p] and status[n_q]):
            v_c = (1.0 - alpha_c)*V[n_p,:] + alpha_c*V[n_q,:]
            dv_c= (1.0 - alpha_c)*gradient_V[n_p,:] + alpha_c*gradient_V[n_q,:]
        elif(status[n_p]):
            v_c = V[n_p,:]
            dv_c= gradient_V[n_p,:]
        elif(status[n_q]):
            v_c = V[n_q,:]
            dv_c= gradient_V[n_q,:]


        # extrapolate to interface, to get v_b
        v_b = v_c + np.dot(x_b-x_c,dv_c)
        # solve half Riemann problem, to get v_b^{Riemann}
        v_bR = _Riemann_bWstar_FS(v_b,vv_wall,nn_wall,eos,self.equation_type)
        # interpolate to edge center, to get v_si
        alpha_si = np.linalg.norm(x_c - x_si)/ np.linalg.norm(x_c - x_b)
        v_si = (1 - alpha_si)*v_c + alpha_si*v_bR

        return v_si


    def _apply_wall_boundary_condition(self,W):
        return ;
'''
