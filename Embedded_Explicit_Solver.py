from Limiter import *
from Intersector import *
from EOS import EOS
import Flux
from Fluid_Domain import *

'''
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
        self.p_oo = V_oo[-1]

        self.gradient_V = np.empty(shape=[fluid_domain.nverts,2,4], dtype = float)






    def _init_fluid_state(self):

        fluid = self.fluid_domain


        W = self.W

        nverts = fluid.nverts

        for i in range(nverts):

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

        self._euler_flux_rhs(V, R)


        self._euler_boundary_flux(V, R)


        #compute diffusion term
        if(self.equation_type == "NS"):

            self._viscid_flux_rhs_fem(V,R)

            self._viscid_boundary_flux_fem(V, R)

    def _lsq_gradient(self,V):

        verts = self.fluid_domain.verts
        nverts = self.fluid_domain.nverts
        connectivity = self.fluid_domain.connectivity

        A = np.empty(shape=[2,2],dtype = float)
        Ainv = np.empty(shape=[2,2],dtype = float)
        b = np.empty(shape=[2,4],dtype = float)

        for i in range(nverts):
            A[:,:] = 0
            b[:,:] = 0
            xc = verts[i,:]
            Vc = V[i,:]

            for neighbor in connectivity[i]:

                dx,dy = verts[neighbor] - xc
                dV = V[neighbor] - Vc

                A[0,0] += dx*dx
                A[0,1] += dx*dy
                A[1,1] += dy*dy
                A[1,0] += dx*dy

                b[0,:] += dx * dV
                b[1,:] += dy * dV

            det =  A[0,0]*A[1,1] - A[1,0]*A[0,1]
            assert np.fabs(det) > 1e-16
            Ainv[0,0],Ainv[0,1],Ainv[1,0],Ainv[1,1] = A[1,1]/det,-A[1,0]/det, -A[0,1]/det,A[0,0]/det

            self.gradient_V[i,:,:] = np.dot(Ainv,b)



    def _euler_flux_rhs(self,V,R):
        # convert conservative state variable W to primitive state variable V

        fluid = self.fluid_domain

        limiter = self.limiter
        eos = self.eos

        self._lsq_gradient(V)

        for i in range(fluid.nedges):


            n,m = fluid.edges[i,:]


            v_n,v_m = V[n,:],V[m,:]



            e_nm = fluid.edge_vector[i,:]

            dr_nm = fluid.directed_area_vector[i,:]

            dv_n,dv_m = np.dot(e_nm,self.gradient_V[n,:,:]),np.dot(e_nm,self.gradient_V[m,:,:])



            v_L, v_R =limiter._reconstruct(v_n, v_m, dv_n, dv_m)

            if(v_L[0] < 0 or v_R[0] < 0):
                print "reconstruct error",i,n,m,v_L,v_R
                print v_n,v_m
                print dv_n,dv_m

            if(v_L[3] < 0 or v_R[3] < 0):
                print "reconstruct error",i,n,m,v_L,v_R
                print v_n,v_m
            flux = Flux._Roe_flux(v_L,v_R,dr_nm,eos)


            R[n,:] -= flux
            R[m,:] += flux



        #R*dt/vol

    def _euler_boundary_flux(self,V,R):

        fluid = self.fluid_domain
        eos = self.eos
        for i in range(fluid.nbounds):

            m,n,e,type = fluid.bounds[i,:]

            x_n = fluid.verts[n,:]

            x_m = fluid.verts[m,:]


            prim_m = V[m,:]

            prim_n = V[n,:]

            dr_nm = 0.5*np.array([x_n[1] - x_m[1], x_m[0] - x_n[0]],dtype=float)


            if(type == 2): #free_stream

                W_oo = self.W_oo

                R[m,:] -= Flux._Steger_Warming(prim_m, W_oo, dr_nm, eos)

                R[n,:] -= Flux._Steger_Warming(prim_n, W_oo, dr_nm, eos)

            elif(type == 3): #slip_wall
                #weakly impose slip_wall boundary condition

                R[m,:] -= [0.0, prim_m[3]*dr_nm[0] , prim_m[3]*dr_nm[1], 0.0]

                R[n,:] -= [0.0, prim_n[3]*dr_nm[0] , prim_n[3]*dr_nm[1], 0.0]

            elif(type == 4): #subsonic_outflow
                # parameterise the outflow state by a freestream pressure rho_f
                p_oo = self.p_oo

                W_moo = np.array([prim_m[0], prim_m[0]*prim_m[1],prim_m[0]*prim_m[2], p_oo/(eos.gamma-1) + 0.5*prim_m[0]*(prim_m[1]**2 + prim_m[2]**2)],dtype=float)

                W_noo = np.array([prim_n[0], prim_n[0]*prim_n[1],prim_n[0]*prim_n[2], p_oo/(eos.gamma-1) + 0.5*prim_n[0]*(prim_n[1]**2 + prim_n[2]**2)],dtype=float)


                R[m,:] -= Flux._Steger_Warming(prim_m, W_moo, dr_nm, eos)

                R[n,:] -= Flux._Steger_Warming(prim_n, W_noo, dr_nm, eos)

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



        for i in range(fluid.nbounds):

            n,m,e,type = fluid.bounds[i,:]


            nx,ny = 0.5*fluid.bound_norm[i,:]

            if(type == 1): # adiabatic wall
                vx_wall, vy_wall, q_flux = fluid.adiabatic_wall

                n1,n2,n3 = fluid.elems[e,:]

                vx1,vy1,T1 = V[n1,1],V[n1,2],eos._compute_temperature(V[n1,:])

                vx2,vy2,T2 = V[n1,1],V[n1,2],eos._compute_temperature(V[n2,:])

                vx3,vy3,T3 = V[n1,1],V[n1,2],eos._compute_temperature(V[n3,:])



                d_shape = self.shape_function_gradient[e,:,:]

                d_v = np.array([  [vx1*d_shape[0,0] + vx2*d_shape[0,1] + vx3*d_shape[0,2],vx1*d_shape[1,0] + vx2*d_shape[1,1] + vx3*d_shape[1,2]]
                                  [vy1*d_shape[0,0] + vy2*d_shape[0,1] + vy3*d_shape[0,2],vy1*d_shape[1,0] + vy2*d_shape[1,1] + vy3*d_shape[1,2]]], dtype=float)

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

        for i in range(fluid.nbounds):

            n,m,e,type = fluid.bounds[i,:]

            if(type == 3): #slip wall
                continue


            if(type == 1): #adiabatic_wall
                vx_wall, vy_wall, T_wall = fluid.bc_cond[type,:]
                W[m,3] += 0.5*W[n,0]*(vx_wall**2 + vy_wall**2 - W[m,1]**2 - W[m,2]**2)
                W[n,3] += 0.5*W[n,0]*(vx_wall**2 + vy_wall**2 - W[n,1]**2 - W[n,2]**2)
                W[n,1:3] = W[n,0]*vx_wall, W[n,0]*vy_wall
                W[m,1:3] = W[m,0]*vx_wall, W[m,0]*vy_wall

            elif(type == 0): #isothermal_wall

                vx_wall, vy_wall, T_wall = fluid.bc_cond[type,:]

                W[n,1:3] = W[n,0]*vx_wall, W[n,0]*vy_wall
                W[m,1:3] = W[m,0]*vx_wall, W[m,0]*vy_wall

                e_wall = T_wall /(eos.gamma*(eos.gamma-1)) + 0.5*vx_wall**2 + 0.5*vy_wall**2
                W[n,3] = W[n,0]*e_wall
                W[m,3] = W[m,0]*e_wall

    def _compute_local_time_step(self,V):

        c = np.sqrt(self.eos.gamma * V[:,3]/ V[:,0])
        wave_speed = np.sqrt(V[:,1]**2 + V[:,2]**2) + c  + self.eos.mu

        dt = self.CFL*self.fluid_domain.min_edge/wave_speed[:,None]

        return dt

    def _compute_residual(self,R,W):

        fluid = self.fluid_domain

        eos = self.eos

        for i in range(fluid.nbounds):

            n,m,e,type = fluid.bounds[i,:]
            #change the flux to the difference between the real wall state and the computed one


            if(type == 0 or type == 1): #adiabatic wall or isothermal wall
                vx_wall, vy_wall = fluid.bc_cond[type,0:2]
                R[m,1] = W[m,1] - W[m,0]*vx_wall
                R[m,2] = W[m,2] - W[m,0]*vy_wall

                R[n,1] = W[n,1] - W[n,0]*vx_wall
                R[n,2] = W[n,2] - W[n,0]*vy_wall



            if(type == 0): #isothermal wall
                vx_wall, vy_wall,T_wall = fluid.bc_cond[type,:]

                e_wall = T_wall /(eos.gamma*(eos.gamma-1)) + 0.5*vx_wall**2 + 0.5*vy_wall**2
                R[m,3] = W[m,3] - W[m,0]*e_wall
                R[n,3] = W[n,3] - W[n,0]*e_wall


        self.res_L2 = np.linalg.norm(R)

        if(np.isinf(self.res_L2_ref)):

            self.res_L2_ref = self.res_L2

        self.res_L2 /= self.res_L2_ref

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
class Embedded_Explicit_Solver(Explicit_Solver):
    def __init__(self,fluid_domain,io_data):
        super().__init__(fluid_domain,io_data)


        self.intersector = Intersector(self.fluid_domain,self.structure)



    def _init_fluid_state(self):

        fluid = self.fluid_domain


        W = self.W

        nverts = fluid.nverts

        for i in range(nverts):

            W[i,:] = self.W_oo

        self._apply_wall_boundary_condition(W)
        self.W = np.load("solutionW.npy")


    def _lsq_gradient(self,V,status):

        verts = self.fluid_domain.verts
        nverts = self.fluid_domain.nverts


        A = np.empty(shape=[2,2],dtype = float)
        Ainv = np.empty(shape=[2,2],dtype = float)
        b = np.empty(shape=[2,4],dtype = float)
        #todo build connectivity


        for i in range(nverts):
            A[:,:] = 0
            b[:,:] = 0
            xc = verts[i,:]
            Vc = V[i,:]

            for neighbor in connectivity[i]:
                if(status[neighbor] ):

                    dx,dy = verts[neighbor] - xc
                    dV = V[neighbor] - Vc

                    A[0,0] += dx*dx
                    A[0,1] += dx*dy
                    A[1,1] += dy*dy
                    A[1,0] += dx*dy

                    b[0,:] += dx * dV
                    b[1,:] += dy * dV

            det =  A[0,0]*A[1,1] - A[1,0]*A[0,1]
            assert np.fabs(det) > 1e-16
            Ainv[0,0],Ainv[0,1],Ainv[1,0],Ainv[1,1] = A[1,1]/det,-A[1,0]/det, -A[0,1]/det,A[0,0]/det

            self.gradient_V[i,:,:] = np.dot(Ainv,b)


















        #self._viscid_flux_rhs_fem(t, with_dt)

    def _euler_flux_rhs(self,V,R):
        # convert conservative state variable W to primitive state variable V

        fluid = self.fluid_domain

        status = self.intersector.status

        intersect_or_not = self.intersector.intersect_or_not

        limiter = self.limiter

        eos = self.eos

        self._lsq_gradient(V,status)

        for i in range(fluid.nedges):
            n,m = fluid.edges[i,:]

            v_n,v_m = V[n,:],V[m,:]

            n_active,m_active = status[n],status[m]

            intersect = intersect_or_not[i]

            e_nm = fluid.edge_vector[i,:]

            dr_nm = fluid.directed_area_vector[i,:]

            dv_n,dv_m = np.dot(e_nm,self.gradient_V[n,:,:]), -np.dot(e_nm,self.gradient_V[m,:,:])

            if(n_active and m_active and not intersect):

                v_L, v_R =limiter._reconstruct(v_n, v_m, dv_n, dv_m)

            elif(n_active and not m_active):

                v_L = v_n + 0.5*dv_n

                v_R = self.SIV(i,self.intersector)

            elif(m_active and not n_active):

                v_R = v_m + 0.5*dv_m

                v_L = self.SIV(i,self.intersector)

            flux = Flux._Roe_flux(v_L,v_R,dr_nm,eos)

            R[n,:] -= flux

            R[m,:] += flux


    def _euler_boundary_flux(self,V,R):

        fluid = self.fluid_domain

        eos = self.eos

        status = self.intersector.status

        for i in range(fluid.nbounds):

            m,n,e,type = fluid.bounds[i,:]

            x_n = fluid.verts[n,:]

            x_m = fluid.verts[m,:]

            status_m = status[m]

            status_n = status[n]

            prim_m = V[m,:]

            prim_n = V[n,:]

            dr_nm = 0.5*np.array([x_n[1] - x_m[1], x_m[0] - x_n[0]],dtype=float)


            if(type == 2): #free_stream

                W_oo = self.W_oo

                if(status_m):
                    R[m,:] -= Flux._Steger_Warming(prim_m, W_oo, dr_nm, eos)
                if(status_n):

                    R[n,:] -= Flux._Steger_Warming(prim_n, W_oo, dr_nm, eos)

            elif(type == 3): #slip_wall
                #weakly impose slip_wall boundary condition
                if(status_m):
                    R[m,:] -= [0.0, prim_m[3]*dr_nm[0] , prim_m[3]*dr_nm[1], 0.0]
                if(status_n):
                    R[n,:] -= [0.0, prim_n[3]*dr_nm[0] , prim_n[3]*dr_nm[1], 0.0]

            elif(type == 4): #subsonic_outflow
                # parameterise the outflow state by a freestream pressure rho_f
                p_oo = self.p_oo
                if(status_m):
                    W_moo = np.array([prim_m[0], prim_m[0]*prim_m[1],prim_m[0]*prim_m[2], p_oo/(eos.gamma-1) + 0.5*prim_m[0]*(prim_m[1]**2 + prim_m[2]**2)],dtype=float)

                    R[m,:] -= Flux._Steger_Warming(prim_m, W_moo, dr_nm, eos)
                if(status_n):

                    W_noo = np.array([prim_n[0], prim_n[0]*prim_n[1],prim_n[0]*prim_n[2], p_oo/(eos.gamma-1) + 0.5*prim_n[0]*(prim_n[1]**2 + prim_n[2]**2)],dtype=float)

                    R[n,:] -= Flux._Steger_Warming(prim_n, W_noo, dr_nm, eos)


    def _fem_ghost_node_update(self, n):
        #update ghost value of node n


        for i in range(fluid.nedges):
            if(intersect_or_not[i]):
                n,m = fluid.edges[i,:]

        return ;

    def _viscid_flux_rhs_fem(self,V,R):
        eos = self.eos
        nelems = self.fluid_domain.nelems
        elems = self.fluid_domain.elems
        shape_function_gradient = self.fluid_domain.shape_function_gradient
        area = self.fluid_domain.area

        tau = np.empty([2, 2], dtype=float)
        F_vis = np.empty(4, dtype=float)
        G_vis = np.empty(4, dtype=float)
        for e in range(nelems):
            n1, n2, n3 = elems[e, :]

            ###########################################
            # ghost value update
            ###################################################
            if(not status[n1] and not status[n2] not status[n3]):
                continue
            for n in elems[e,:]:
                if(not status[n]):
                    V[n,:] = self._fem_ghost_node_update(n)

            #########################
            # Compute velocity gradients, temperature gradients, store in d_v
            #####################


            vx1, vy1, T1 = V[n1, 1], V[n1, 2], eos._compute_temperature(V[n1, :])
            vx2, vy2, T2 = V[n2, 1], V[n2, 2], eos._compute_temperature(V[n2, :])
            vx3, vy3, T3 = V[n3, 1], V[n3, 2], eos._compute_temperature(V[n3, :])

            vx, vy, T = (vx1 + vx2 + vx3) / 3.0, (vy1 + vy2 + vy3) / 3.0, (T1 + T2 + T3) / 3.0

            d_shape = shape_function_gradient[e, :, :]

            d_v = np.array([[vx1 * d_shape[0, 0] + vx2 * d_shape[0, 1] + vx3 * d_shape[0, 2],
                             vx1 * d_shape[1, 0] + vx2 * d_shape[1, 1] + vx3 * d_shape[1, 2]],
                            [vy1 * d_shape[0, 0] + vy2 * d_shape[0, 1] + vy3 * d_shape[0, 2],
                             vy1 * d_shape[1, 0] + vy2 * d_shape[1, 1] + vy3 * d_shape[1, 2]],
                            [T1 * d_shape[0, 0] + T2 * d_shape[0, 1] + T3 * d_shape[0, 2],
                             T1 * d_shape[1, 0] + T2 * d_shape[1, 1] + T3 * d_shape[1, 2]]], dtype=float)

            ########################################################
            # Test dante's dT/dx and dT/dy
            #######################################################
            '''
            rho1,rho2,rho3 = V[n1,0], V[n2,0], V[n3,0]
            p1, p2, p3 = V[n1,3], V[n2,3], V[n3,3]
            rho = (rho1 + rho2 + rho3)/3.0
            p = (p1 + p2 + p3)/3.0
            dT_drho, dT_dp = -eos.gamma*p/(rho**2), eos.gamma/rho
            drho_dx,drho_dy = rho1*d_shape[0,0] + rho2*d_shape[0,1] + rho3*d_shape[0,2], rho1*d_shape[1,0] + rho2*d_shape[1,1] + rho3*d_shape[1,2]
            dp_dx, dp_dy =    p1*d_shape[0,0] +   p2*d_shape[0,1]   + p3*d_shape[0,2],   p1*d_shape[1,0] +   p2*d_shape[1,1] +   p3*d_shape[1,2]
            dT_dx, dT_dy = dT_drho *drho_dx + dT_dp*dp_dx, dT_drho *drho_dy + dT_dp*dp_dy

            d_v[2,:] = [dT_dx, dT_dy]
            '''

            #############################################
            # compute viscosity, heat conductivity
            ############################################


            vis_mu, vis_lambda, tem_k = eos._transport_coefficients()

            ############################
            # Compute stress tensor, store in tau
            ############################

            tau[0, 0] = 2 * vis_mu * d_v[0, 0] + vis_lambda * (d_v[0, 0] + d_v[1, 1])
            tau[0, 1] = vis_mu * (d_v[0, 1] + d_v[1, 0])
            tau[1, 1] = 2 * vis_mu * d_v[1, 1] + vis_lambda * (d_v[0, 0] + d_v[1, 1])
            tau[1, 0] = tau[0, 1]

            #####################
            # Compute viscous flux
            ######################
            F_vis[:] = 0, -tau[0, 0], -tau[0, 1], -tau[0, 0] * vx - tau[0, 1] * vy - tem_k * d_v[2, 0]
            G_vis[:] = 0, -tau[0, 1], -tau[1, 1], -tau[0, 1] * vx - tau[1, 1] * vy - tem_k * d_v[2, 1]

            e_area = area[e]
            R[n1, :] += e_area * d_shape[0, 0] * F_vis + e_area * d_shape[1, 0] * G_vis
            R[n2, :] += e_area * d_shape[0, 1] * F_vis + e_area * d_shape[1, 1] * G_vis
            R[n3, :] += e_area * d_shape[0, 2] * F_vis + e_area * d_shape[1, 2] * G_vis





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
        v_bR = Flux._Riemann_bWstar_FS(v_b,vv_wall,nn_wall,eos,self.equation_type)
        # interpolate to edge center, to get v_si
        alpha_si = np.linalg.norm(x_c - x_si)/ np.linalg.norm(x_c - x_b)
        v_si = (1 - alpha_si)*v_c + alpha_si*v_bR

        return v_si


    def _apply_wall_boundary_condition(self,W):
        return ;
