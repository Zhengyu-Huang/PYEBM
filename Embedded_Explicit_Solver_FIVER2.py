from Limiter import *
from Intersector import *
from EOS import EOS
import Flux
from Explicit_Solver import Explicit_Solver
from Fluid_Domain import *


def _interpolation(x_1, y_1, x_2, y_2, x_3):

    d = (x_2[0] - x_1[0])**2 + (x_2[1] - x_1[1])**2

    assert(d > 1e-10)

    alpha = (x_3[0] - x_1[0])**2 + (x_3[1] - x_1[1])**2 / d

    l = len(y_1)

    return [y_1[i] + alpha*(y_2[i] - y_1[i]) for i in range(l)]


#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
class Embedded_Explicit_Solver(Explicit_Solver):
    def __init__(self, fluid_domain, structure, io_data):
        """
        Initialize Embedded Explicit Solver

        Args:
            fluid_domain:  Fluid_Domain class
            structure:     Structure class
            io_data:       Input information

        Attributes:
            intersector: save fluid embedded surface intersection information
            ghost_node: save ghost node variables vx, vy, T ; vx , vy, T the first ghost value is on the same side
            with the fluid node(interpolating with the first ghost_node_stencil);
            the second ghost value is on the other side(interpolating with the first ghost_node_stencil);
        """
        super().__init__(fluid_domain,  io_data)
        self.structure = structure
        self.intersector = Intersector(self.fluid_domain, self.structure)
        #self.ghost_nodes = np.empty(shape=[self.fluid_domain.nverts, 8],dtype=float)
        self.ghost_nodes = np.empty(shape=[self.fluid_domain.nverts, 6], dtype=float)



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



    def _fem_ghost_node_update(self, V):
        # update ghost value of node n
        nverts = self.fluid_domain.nverts
        verts = self.fluid_domain.verts

        ghost_node_stencil = self.intersector.ghost_node_stencil
        ghost_node_closest_position = self.intersector.ghost_node_closest_position

        ghost_nodes = self.ghost_nodes

        for n in range(nverts):
            for side in range(2):
                n_p,n_q,beta = ghost_node_stencil[n][3*side],ghost_node_stencil[n][3*side+1], ghost_node_stencil[n][3*side+2]
                if(n_p < 0 and n_q < 0):
                    continue

                x_n = verts[n]

                x_p = verts[n_p]

                x_q = verts[n_q]

                x_c = (1 - beta)*x_p + beta*x_q


                xs_info , s_alpha= ghost_node_closest_position[n]

                x_s, vv_wall, nn_wall = self.structure._point_info(xs_info, s_alpha)

                T_wall = self.structure._temperature(xs_info,s_alpha)


                if(n_p >= 0 and n_q >= 0):

                    v_c = (1.0 - beta) * V[n_p, :] + beta * V[n_q, :]

                    dv_c = (1.0 - beta) * self.gradient_V[n_p, :] + beta * self.gradient_V[n_q, :]

                else:

                    v = V[n_p, :] if n_p > -1 else V[n_q, :]

                    dv = self.gradient_V[n_p, :] if n_p > -1 else self.gradient_V[n_q, :]

                    v_c = v + np.dot(x_n - x_c, dv)

                    dv_c = dv


                u_c = [v_c[1],v_c[2],self.eos._compute_temperature(v_c)]
                u_wall = [vv_wall[0],vv_wall[1],T_wall]
                ghost_nodes[n, 3*side:3*side+3] = _interpolation(x_c, u_c, x_s, u_wall, x_n)


                '''
                #todo dante uses high order interpolation in IB_drive:f90:set_FEM_points

                eta = np.linalg.norm(x_n - x_c)
                xi = np.linalg.norm(x_s - x_c)
                Dr = (x_s-x_c)/xi
                g =[v_c[0],0.,0.,0.]
                coeff_3 = v_c[1]
                coeff_2 = np.dot(dv_c[:,1],Dr)
                coeff_1 = (vv_wall[0] - coeff_3 - coeff_2*xi)/xi**2
                g[1] = coeff_1*eta**2 + coeff_2*eta + coeff_3

                coeff_3 = v_c[2]
                coeff_2 = np.dot(dv_c[:, 2], Dr)
                coeff_1 = (vv_wall[1] - coeff_3 - coeff_2 * xi) / xi ** 2
                g[2] = coeff_1 * eta ** 2 + coeff_2 * eta + coeff_3

                coeff_3 = self.eos._compute_temperature(v_c)
                dT_drho, dT_dp = -self.eos.gamma * v_c[3] / (v_c[0] ** 2), self.eos.gamma / v_c[0]
                dT_dx, dT_dy = dT_drho * dv_c[:, 0] + dT_dp * dv_c[:, 3]
                coeff_2 = dT_dx *Dr[0] + dT_dy*Dr[1]
                coeff_1 = (T_wall - coeff_3 - coeff_2 * xi) / xi ** 2
                ghost_T = coeff_1 * eta ** 2 + coeff_2 * eta + coeff_3
                g[3] = ghost_T *g[0]/self.eos.gamma

                ghost_nodes[n, 4 * side:4 * side + 4] = g
                '''





        return;



    def _viscid_flux_rhs_fem(self, V, R):
        eos = self.eos
        nelems = self.fluid_domain.nelems
        elems = self.fluid_domain.elems
        elem_edge_neighbors = self.fluid_domain.elem_edge_neighbors
        shape_function_gradient = self.fluid_domain.shape_function_gradient
        area = self.fluid_domain.area
        edges = self.fluid_domain.edges
        intersect_or_not = self.intersector.intersect_or_not
        status = self.intersector.status



        self._fem_ghost_node_update(V)
        ghost_nodes = self.ghost_nodes


        tau = np.empty([2, 2], dtype=float)
        F_vis = np.empty(4, dtype=float)
        G_vis = np.empty(4, dtype=float)

        ele_V = np.empty([3,3],dtype=float)
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
                    m1,m2 = edges[i]
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

            '''
            ele_V2 = np.empty([3, 4], dtype=float)
            for local_i in range(3):
                n = elems[e, local_i]
                if (status[n]):
                    ele_V2[local_i, :] = V[n, :]
                else:

                    if n in inactive_side:
                        ele_V2[local_i, :] = ghost_nodes[n, 4:8]
                    else:
                        ele_V2[local_i, :] = ghost_nodes[n, 0:4]
            '''






            #########################
            # Compute velocity gradients, temperature gradients, store in d_v
            #####################


            vx1, vy1, T1 = ele_V[0, :]
            vx2, vy2, T2 = ele_V[1, :]
            vx3, vy3, T3 = ele_V[2, :]

            '''
            vx1, vy1, T1 = ele_V2[0, 1],ele_V2[0, 2], self.eos._compute_temperature(ele_V2[0,:])
            vx2, vy2, T2 = ele_V2[1, 1],ele_V2[1, 2], self.eos._compute_temperature(ele_V2[1,:])
            vx3, vy3, T3 = ele_V2[2, 1],ele_V2[2, 2], self.eos._compute_temperature(ele_V2[2,:])
            '''


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
            rho1,rho2,rho3 = ele_V2[0,0], ele_V2[1,0], ele_V2[2,0]
            p1, p2, p3 = ele_V2[0,3], ele_V2[1,3], ele_V2[2,3]
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





    def _compute_local_time_step(self, V):

        c = np.sqrt(self.eos.gamma * V[:, 3] / V[:, 0])
        wave_speed = np.sqrt(V[:, 1] ** 2 + V[:, 2] ** 2) + c + self.eos.mu

        dt = self.CFL * self.fluid_domain.min_edge / wave_speed[:, None]

        return dt




