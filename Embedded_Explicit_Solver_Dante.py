from Limiter import *
from Intersector_Dante import *
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
        self.ghost_nodes = np.empty(shape=[self.fluid_domain.nverts, 6], dtype=float)







    def FIVER(self, V, i, side, limiter = None):
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
        x_s, vv_wall, nn_wall = self.structure._point_info(struc_i, struc_alpha, x_b)

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

                x_s, vv_wall, nn_wall = self.structure._point_info(xs_info, s_alpha, x_c)

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
                ghost_nodes[n, 3*side:3*side+3] = Utility._interpolation(x_c, u_c, x_s, u_wall, x_n)






        return;


    def _ghost_value(self, n, opposite_side):
        '''
        :param n: int, ghost node number
        :param opposite_side: bool , use the ghost value on the different side of structure or not
        :return: u,v,T of the ghost node
        '''

        if(opposite_side):
            return self.ghost_nodes[n,3:6]
        else:
            return self.ghost_nodes[n, 0:3]



