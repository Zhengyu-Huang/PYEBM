from Limiter import *
from Intersector import *
from EOS import EOS
import Flux
from Explicit_Solver import Explicit_Solver
from Fluid_Domain import *



#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
class Embedded_Explicit_Solver(Explicit_Solver):
    def __init__(self, fluid_domain, structure, io_data):
        super().__init__(fluid_domain,  io_data)
        self.structure = structure
        self.intersector = Intersector(self.fluid_domain, self.structure)



    def _lsq_gradient(self, V, status):

        verts = self.fluid_domain.verts
        nverts = self.fluid_domain.nverts
        connectivity = self.intersector.connectivity

        A = np.empty(shape=[2, 2], dtype=float)
        Ainv = np.empty(shape=[2, 2], dtype=float)
        b = np.empty(shape=[2, 4], dtype=float)


        for i in range(nverts):
            if not status[i]:
                continue
            A[:, :] = 0
            b[:, :] = 0
            xc = verts[i, :]
            Vc = V[i, :]


            for neighbor in connectivity[i]:
                if not status[neighbor]:
                    continue

                dx, dy = verts[neighbor] - xc
                dV = V[neighbor] - Vc

                A[0, 0] += dx * dx
                A[0, 1] += dx * dy
                A[1, 1] += dy * dy
                A[1, 0] += dx * dy

                b[0, :] += dx * dV
                b[1, :] += dy * dV

            det = A[0, 0] * A[1, 1] - A[1, 0] * A[0, 1]
            assert  np.fabs(det) > 1e-16
            Ainv[0, 0], Ainv[0, 1], Ainv[1, 0], Ainv[1, 1] = A[1, 1] / det, -A[1, 0] / det, -A[0, 1] / det, A[
                0, 0] / det

            self.gradient_V[i, :, :] = np.dot(Ainv, b)



    def _euler_flux_rhs(self, V, R):
        # convert conservative state variable W to primitive state variable V

        fluid = self.fluid_domain

        status = self.intersector.status

        intersect_or_not = self.intersector.intersect_or_not

        limiter = self.limiter

        eos = self.eos

        self._lsq_gradient(V, status)

        for i in range(fluid.nedges):

            n, m = fluid.edges[i, :]
            if(n == 2484 or m == 2484):
                print('stop')

            v_n, v_m = V[n, :], V[m, :]

            n_active, m_active = status[n], status[m]

            intersect = intersect_or_not[i]

            e_nm = fluid.edge_vector[i, :]

            dr_nm = fluid.directed_area_vector[i, :]

            dv_n, dv_m = np.dot(e_nm, self.gradient_V[n, :, :]), -np.dot(e_nm, self.gradient_V[m, :, :])

            if (n_active and m_active and not intersect):

                v_L, v_R = limiter._reconstruct(v_n, v_m, dv_n, dv_m)

            elif (n_active and not m_active):

                v_L = v_n + 0.5 * dv_n #todo limiter

                v_R = self.SIV(V, i,limiter)

            elif (m_active and not n_active):

                v_R = v_m + 0.5 * dv_m #todo limiter

                v_L = self.SIV(V, i,limiter)

            flux = Flux._Roe_flux(v_L, v_R, dr_nm, eos)

            if n_active:
                R[n, :] -= flux
            if m_active:
                R[m, :] += flux

    def SIV(self, V, i, limiter = None):
        # p---c----q
        #    \
        #     \
        #  n1  b  n2
        #      \
        #       \
        #        s

        verts = self.fluid_domain.verts

        gradient_V = self.gradient_V

        n1, n2 = self.fluid_domain.edges[i, :]

        eos = self.eos

        status = self.intersector.status

        x_b = 0.5 * (verts[n1, :] + verts[n2, :])

        n_p, n_q, alpha_c = self.intersector.edge_center_stencil[i]

        # Information of closest point on the structure
        struc_i, struc_alpha = self.intersector.edge_center_closest_position[i]
        x_s, vv_wall, nn_wall = self.structure._point_info(struc_i, struc_alpha)

        # Construct x_c information, including primitive variable v_c, dv_c
        x_p, x_q = verts[n_p, :], verts[n_q, :]
        x_c = (1 - alpha_c) * x_p + alpha_c * x_q
        if (status[n_p] and status[n_q]):
            v_c = (1.0 - alpha_c) * V[n_p, :] + alpha_c * V[n_q, :]
            dv_c = (1.0 - alpha_c) * gradient_V[n_p, :] + alpha_c * gradient_V[n_q, :]
        elif (status[n_p]):

            v_c = V[n_p, :]
            dv_c = gradient_V[n_p, :]
        elif (status[n_q]):

            v_c = V[n_q, :]
            dv_c = gradient_V[n_q, :]

        # extrapolate to interface, to get v_b
        v_s = v_c + np.dot(x_s - x_c, dv_c) #todo limiter
        # solve half Riemann problem, to get v_b^{Riemann}
        v_bR = Flux._Riemann_bWstar_FS(v_s, vv_wall, nn_wall, eos, self.equation_type)
        # interpolate to edge center, to get v_si
        alpha_b = np.linalg.norm(x_c - x_b) / np.linalg.norm(x_c - x_s)

        v_b = (1 - alpha_b) * v_c + alpha_b * v_bR

        return v_b



    def _fem_ghost_node_update(self, V, num_side):
        # update ghost value of node n
        nverts = self.nverts
        verts = self.verts

        ghost_node_stencil = self.intersector.ghost_node_stencil
        ghost_node_closest_points = self.intersector.ghost_node_closest_position

        for n in range(nverts):
            for side in range(num_side):
                n_p,n_q,beta = ghost_node_stencil[n][3*side],ghost_node_stencil[n][3*side+1], ghost_node_stencil[n][3*side+2]
                if(n_p < 0 and n_q < 0):
                    continue

                x_b = verts[n]

                x_p = verts[n_p]

                x_q = verts[n_q]

                x_c = (1 - beta)*x_p + beta*x_q


                xs_info = ghost_node_closest_points[n]

                x_s, vv_wall, nn_wall = self.structure._point_info(xs_info)

                T_wall = self.structure._temperature(xs_info)


                if(n_p >= 0 and n_q >= 0):

                    v_c = (1.0 - beta) * V[n_p, :] + beta * V[n_q, :]

                else:

                    v = V[n_p, :] if n_p > -1 else V[n_q, :]

                    dv = gradient_V[n_p, :] if n_p > -1 else gradient_V[n_q, :]

                    v_c = v + np.dot(x_b - x_c, dv)


                ghost_node[n, 3*side:3*side+3] = _ghost_interpolation(x_c, v_c,  x_b,  x_s, vv_wall,T_wall)
                #todo dante uses high order interpolation in IB_drive:f90:set_FEM_points

        return;

    def _ghost_interpolation(x_c, v_c,  x_b,  x_s, vv_wall, T_wall):
        vx_c,vy_c,T_c = v_c[1],v_c[2], eos._compute_temperature(v_c)
        vx_s,vy_s,T_s = vv_wall[0],vv_wall[1], T_wall

        alpha = np.dot(x_c - x_s,x_b - x_s)/np.dot(x_c - x_s,x_c - x_s)


        return vx_s + alpha*(vx_c - vx_s),vy_s + alpha*(vy_c - vy_s), T_s + alpha*(T_c - T_s)

    def _viscid_flux_rhs_fem(self, V, R):
        eos = self.eos
        nelems = self.fluid_domain.nelems
        elems = self.fluid_domain.elems
        elem_edge_neighbors = self.fluid_domain.elem_edge_neighbors
        shape_function_gradient = self.fluid_domain.shape_function_gradient
        intersect_or_not = self.intersector.intersect_or_not
        status = self.intersector.status
        ghost_nodes = self.ghost_nodes
        area = self.fluid_domain.area

        tau = np.empty([2, 2], dtype=float)
        F_vis = np.empty(4, dtype=float)
        G_vis = np.empty(4, dtype=float)

        ele_V = np.empty([3,4],dtype=float)
        for e in range(nelems):

            n1, n2, n3 = elems[e, :]

            ###########################################
            # ghost value update
            ###################################################
            if (not status[n1] and not status[n2] and not status[n3]):
                continue
            ##################################################
            # decide which ghost value to use
            ##################################################
            inactive_side = set()

            for i in elem_edge_neighbors[e]:
                    if(intersect_or_not[i]):
                        for m1,m2 in edges[i]:
                            if status[m1] and not status[m2]:
                                inactive_side.add(m2)
                            elif status[m2] and not status[m1]:
                                inactive_side.add(m1)

            for local_i in range(3):
                n = elems[e, local_i]
                if (status[n]):
                    ele_V[local_i, :] = V[n, 1], V[n, 2], eos._compute_temperature(V[n, :])
                else:

                    if n in inactive_side:
                        ele_V[local_i, :] = ghost_nodes[n,3:6]
                    else:
                        ele_V[local_i, :] = ghost_nodes[n,0:3]







            #########################
            # Compute velocity gradients, temperature gradients, store in d_v
            #####################


            vx1, vy1, T1 = ele_V[0, :]
            vx2, vy2, T2 = ele_V[1, :]
            vx3, vy3, T3 = ele_V[2, :]

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


    def _apply_wall_boundary_condition(self, W):

        fluid = self.fluid_domain

        eos = self.eos

        for i in range(fluid.nbounds):

            n, m, e, type = fluid.bounds[i, :]

            if (type == 3):  # slip wall
                continue

            if (type == 1):  # adiabatic_wall
                vx_wall, vy_wall, T_wall = fluid.bc_cond[type, :]
                W[m, 3] += 0.5 * W[n, 0] * (vx_wall ** 2 + vy_wall ** 2 - W[m, 1] ** 2 - W[m, 2] ** 2)
                W[n, 3] += 0.5 * W[n, 0] * (vx_wall ** 2 + vy_wall ** 2 - W[n, 1] ** 2 - W[n, 2] ** 2)
                W[n, 1:3] = W[n, 0] * vx_wall, W[n, 0] * vy_wall
                W[m, 1:3] = W[m, 0] * vx_wall, W[m, 0] * vy_wall

            elif (type == 0):  # isothermal_wall

                vx_wall, vy_wall, T_wall = fluid.bc_cond[type, :]

                W[n, 1:3] = W[n, 0] * vx_wall, W[n, 0] * vy_wall
                W[m, 1:3] = W[m, 0] * vx_wall, W[m, 0] * vy_wall

                e_wall = T_wall / (eos.gamma * (eos.gamma - 1)) + 0.5 * vx_wall ** 2 + 0.5 * vy_wall ** 2
                W[n, 3] = W[n, 0] * e_wall
                W[m, 3] = W[m, 0] * e_wall

    def _compute_local_time_step(self, V):

        c = np.sqrt(self.eos.gamma * V[:, 3] / V[:, 0])
        wave_speed = np.sqrt(V[:, 1] ** 2 + V[:, 2] ** 2) + c + self.eos.mu

        dt = self.CFL * self.fluid_domain.min_edge / wave_speed[:, None]

        return dt




