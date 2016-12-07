from Limiter import *
from Intersector import *

import Flux
from Explicit_Solver import Explicit_Solver
from Fluid_Domain import *



#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
class Embedded_Explicit_Solver_Base(Explicit_Solver):
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

                flux = Flux._Roe_flux(v_L, v_R, dr_nm, eos)

                R[n, :] -= flux

                R[m, :] += flux

            else:
                #############################################################
                # consider there are two case,
                #
                #
                #
                #
                #############################################################

                if (n_active):

                    v_L = v_n + 0.5 * dv_n #todo limiter

                    v_R = self.FIVER(V, i, n, limiter)

                    flux = Flux._Roe_flux(v_L, v_R, dr_nm, eos)

                    R[n, :] -= flux

                if (m_active):

                    v_R = v_m + 0.5 * dv_m #todo limiter

                    v_L = self.FIVER(V, i, m, limiter)

                    flux = Flux._Roe_flux(v_L, v_R, dr_nm, eos)

                    R[m, :] += flux





    def FIVER(self, V, i, n, limiter = None):
        '''
        Impose transmission condition

        :param V: primitive state variables
        :param i: edge number, edge i has one ghost node and one active node,
        or intersected by embedded surface
        :param limiter: flow limiter
        :param n: compute FIVER flux for node n, do everything on the node n side
        :return:

        Return the primitive state variables on ghost side of the edge.
        '''

        return ;



    def _fem_ghost_node_update(self, V):
        '''
        Virtual function, compute the primitive state variables for ghost nodes
        :param V: primitive state variables
        :return:
        '''
        return;

    def _ghost_value(self, n, opposite_side):
        '''
        :param n: int, ghost node number
        :param opposite_side: bool , use the ghost value on the different side of structure or not
        :return: u,v,T of the ghost node
        '''

    def _viscid_flux_elem(self, elem_uvT, d_shape, area):
        '''
        compute viscous flux in the one element
        :param elem_uvT: 3 by 3 matrix, each row stores u,v ,T of one node
        :param d_shape: 2 by 3 matrix, we have three basis functions phi1, phi2 and phi3, first line stores its x derivative
               and second line store its y derivation
        :return: the flux components for 3 nodes(to add to flux, you should multiplies area)
        '''
        eos = self.eos

        tau = np.empty([2, 2], dtype=float)
        F_vis = np.empty(4, dtype=float)
        G_vis = np.empty(4, dtype=float)

        vx1, vy1, T1 = elem_uvT[0, :]
        vx2, vy2, T2 = elem_uvT[1, :]
        vx3, vy3, T3 = elem_uvT[2, :]

        vx, vy, T = (vx1 + vx2 + vx3) / 3.0, (vy1 + vy2 + vy3) / 3.0, (T1 + T2 + T3) / 3.0

        d_v = np.array([[vx1 * d_shape[0, 0] + vx2 * d_shape[0, 1] + vx3 * d_shape[0, 2],
                         vx1 * d_shape[1, 0] + vx2 * d_shape[1, 1] + vx3 * d_shape[1, 2]],
                        [vy1 * d_shape[0, 0] + vy2 * d_shape[0, 1] + vy3 * d_shape[0, 2],
                         vy1 * d_shape[1, 0] + vy2 * d_shape[1, 1] + vy3 * d_shape[1, 2]],
                        [T1 * d_shape[0, 0] + T2 * d_shape[0, 1] + T3 * d_shape[0, 2],
                         T1 * d_shape[1, 0] + T2 * d_shape[1, 1] + T3 * d_shape[1, 2]]], dtype=float)

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

        return area * d_shape[0, 0] * F_vis + area * d_shape[1, 0] * G_vis, \
               area * d_shape[0, 1] * F_vis + area * d_shape[1, 1] * G_vis, \
               area * d_shape[0, 2] * F_vis + area * d_shape[1, 2] * G_vis

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



        for e in range(nelems):

            n1, n2, n3 = elems[e, :]

            elem_uvT = np.empty(shape=[3, 3], dtype=float)



            if (not status[n1] and not status[n2] and not status[n3]):
                ###########################################
                # all nodes are ghost nodes
                ###################################################
                continue
            elif (status[n1] and status[n2] and status[n3]):
                ###########################################
                # all nodes are active nodes
                ###################################################
                e_area = area[e]
                d_shape = shape_function_gradient[e, :, :]

                elem_uvT[0, :] = V[n1, 1], V[n1, 2], eos._compute_temperature(V[n1, :])
                elem_uvT[1, :] = V[n2, 1], V[n2, 2], eos._compute_temperature(V[n2, :])
                elem_uvT[2, :] = V[n3, 1], V[n3, 2], eos._compute_temperature(V[n3, :])

                R1, R2, R3 = self._viscid_flux_elem(elem_uvT, d_shape, e_area)

                R[n1, :] += R1
                R[n2, :] += R2
                R[n3, :] += R3
                '''
                if (n1 == 1643 or n2 == 1643 or n3 == 1643):
                    print(n1, n2, n3, 'compute ', 'flux is', R1, R2, R3)
                    print(elem_uvT)
                '''
            else:
                ##################################################
                # some nodes are ghost but some are not
                ##################################################
                for i in range(3):
                    ni = elems[e, i]
                    if not status[ni]:
                        continue
                    elem_uvT[i,:] = V[ni, 1], V[ni, 2], eos._compute_temperature(V[ni, :])
                    ################################################################################################
                    #if j is ghost  and (ij) is intersected, use ghost value on the other side of the structure
                    #               and (ij) is not intersected, use ghost value on the same side of the structure
                    #if j is active and (ij) is not intersected,  use its real value
                    #               and (ij) is intersected, use ghost value on the other side of the structure
                    ################################################################################################

                    for j in range(3):
                        if j == i:
                            continue
                        nj = elems[e,j]
                        for ij in elem_edge_neighbors[e]:
                            m1, m2 = edges[ij]
                            if(m1 == ni and m2 == nj) or (m1 == nj and m2 == ni):
                                ij_intersect = intersect_or_not[ij]
                                break
                        if(ij_intersect):
                            elem_uvT[j, :] = self._ghost_value(nj,True)
                        elif status[nj]:
                            elem_uvT[j, :] = V[nj, 1], V[nj, 2], eos._compute_temperature(V[nj, :])
                        else:
                            elem_uvT[j, :] = self._ghost_value(nj, False)
                    '''

                    inactive_side = set()

                    for l in elem_edge_neighbors[e]:
                        if (intersect_or_not[l]):
                            m1, m2 = edges[l]
                            if status[m1] and not status[m2]:
                                inactive_side.add(m2)
                            elif status[m2] and not status[m1]:
                                inactive_side.add(m1)

                    for local_i in range(3):
                        n = elems[e, local_i]
                        if(n ==ni):
                            continue
                        else:

                            if n in inactive_side:
                                elem_uvT[local_i, :] = self._ghost_value(n, True)
                            else:
                                elem_uvT[local_i, :] = self._ghost_value(n, False)


                    '''


                    e_area = area[e]
                    d_shape = shape_function_gradient[e, :, :]
                    ele_flux = self._viscid_flux_elem(elem_uvT, d_shape, e_area)

                    R[ni, :] += ele_flux[i]












