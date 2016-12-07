from Limiter import *
from Intersector_FIVER2 import *
import Utility
import Flux
from Embedded_Explicit_Solver_Base import Embedded_Explicit_Solver_Base
from Fluid_Domain import *





#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
class Embedded_Explicit_Solver(Embedded_Explicit_Solver_Base):
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
        super().__init__(fluid_domain,  structure, io_data)

        self.intersector = Intersector(self.fluid_domain, self.structure)
        #self.ghost_nodes = np.empty(shape=[self.fluid_domain.nverts, 8],dtype=float)
        self.ghost_nodes = np.empty(shape=[self.fluid_domain.nverts, 3], dtype=float)







    def FIVER(self, V, i, n, limiter = None, FIVER_order = 1):
        #  x1         xc  xb     x2
        #                  \
        #                   \
        #                    s

        verts = self.fluid_domain.verts

        status = self.intersector.status

        gradient_V = self.gradient_V

        n1, n2 = self.fluid_domain.edges[i, :]

        m = n1 if n2 == n else n2

        x1, x2 = verts[n, :], verts[m, :]

        assert(status[n])

        eos = self.eos




        x_c = 0.5 * (verts[n1, :] + verts[n2, :])

        alpha_1, s_1, beta_1, alpha_2, s_2, beta_2 = self.intersector.intersect_result[i]

        alpha, s, beta = (alpha_1, s_1,beta_1) if n == n1 else (alpha_2, s_2, beta_2)




        # Information of closest point on the structure
        x_s, vv_wall, nn_wall = self.structure._point_info(s, beta, x1)

        # Construct x_c information, including primitive variable v_c, dv_c
        x_b = (1 - alpha) * x1 + alpha * x2

        v = V[n, :]

        dv = gradient_V[n,:]

        v_b = v + np.dot(x_b - x1, dv)


        v_bR = Flux._Riemann_bWstar_FS(v_b, vv_wall, nn_wall, eos, self.equation_type)



        #The following two lines correspond to first order FIVER and second order FIVER
        if(FIVER_order == 1):
            v_c = v_bR                 # first order FIVER
        else:
            v_c = Utility._interpolation(x1, v, x_b, v_bR, x_c)   # second order FIVER


        #print (v_c)

        return v_c



    def _fem_ghost_node_update(self, V):
        # update ghost value of all nodes
        nverts = self.fluid_domain.nverts

        verts = self.fluid_domain.verts

        status = self.intersector.status

        intersect_or_not = self.intersector.intersect_or_not

        nedges = self.fluid_domain.nedges

        edges = self.fluid_domain.edges

        ghost_nodes = self.ghost_nodes

        ghost_nodes[:] = 0

        weight = np.zeros(nverts,dtype=int)

        for i in range(nedges):
            if not intersect_or_not[i]:
                continue
            n,m = edges[i,:]
            if not status[n] and not status[m]:
                continue

            ###########################################################
            # now it is intersected and at least one node is active
            #########################################################
            xn,xm = verts[n,:], verts[m,:]

            alpha_1, s_1, beta_1, alpha_2, s_2, beta_2 = self.intersector.intersect_result[i]

            if status[n]:
                x_b = (1 - alpha_1) * xn + alpha_1 * xm

                x_s, vv_wall, nn_wall = self.structure._point_info(s_1, beta_1, xn)

                T_wall = self.structure._temperature(s_1, beta_1)

                vn = V[n,:]

                un = [vn[1], vn[2], self.eos._compute_temperature(vn)]

                u_wall = [vv_wall[0], vv_wall[1], T_wall]

                ghost_nodes[m, :] += Utility._interpolation(xn, un, x_b, u_wall, xm)

                weight[m] += 1

            if status[m]:
                x_b = (1 - alpha_2) * xm + alpha_2 * xn

                x_s, vv_wall, nn_wall = self.structure._point_info(s_2, beta_2, xm)

                T_wall = self.structure._temperature(s_2, beta_2)

                vm = V[m, :]

                um = [vm[1], vm[2], self.eos._compute_temperature(vm)]

                u_wall = [vv_wall[0], vv_wall[1], T_wall]

                ghost_nodes[n, :] += Utility._interpolation(xm, um, x_b, u_wall, xn)

                weight[n] += 1



        for n in range(nverts):
            if weight[n]  > 0 :
                ghost_nodes[n,:] /= weight[n]




        return;


    def _ghost_value(self, n, opposite_side):
        '''
        :param n: int, ghost node number
        :param opposite_side: bool , use the ghost value on the different side of structure or not
        :return: u,v,T of the ghost node
        '''


        return self.ghost_nodes[n,:]






