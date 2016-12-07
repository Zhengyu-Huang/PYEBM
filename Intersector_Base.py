import numpy as np
import copy as copy
import matplotlib.pyplot as plt
from Fluid_Domain import Fluid_Domain
from Structure import Structure


def _box_intersect(box1, box2):
    """
    lazy intersection box intersection

    Args:
        box1: a list of 4 float, mincorner: box[0], box[1] , maxcorner: box[2], box[3]
        box2: a list of 4 float, mincorner: box[0], box[1] , maxcorner: box[2], box[3]

    Returns:
        bool: True, if two boxes lazily intersect, else False

    """
    return (box1[0] <= box2[2] and box1[1] <= box2[3] and box2[0] <= box1[2] and box2[1] <= box1[3])


def _box_candidate(box1, boxes, candidates):
    """
    find box in boxes which lazily intersects box1

    Args:
        box1: a list of 4 float, mincorner: box[0], box[1] , maxcorner: box[2], box[3]
        boxes: a list of box(4 float box)
        candidates: an empty list of int

    Returns:
        bool: True, if there is box in boxes lazily intersect box1, else False
        append the their box ids, which lazily intersect box1, to candidates

    """

    i = 0;
    for x1, y1, x2, y2 in boxes:
        if (box1[0] <= x2 and box1[1] <= y2 and x1 <= box1[2] and y1 <= box1[3]):
            candidates.append(i)
        i += 1
    return False if not candidates else True


def _compute_dist(p, x1, x2):
    """
    compute the distance between vertex p, and segment (x1,x2), the closest point x1 + alpha*(x2-x1)

    Args:
        p:  2d numpy vector
        x1: 2d numpy vector
        x2: 2d numpy vector

    Returns:
        the distance between vertex p, and segment (x1,x2) and closest point location alpha
    """

    cross = np.dot(x2 - x1, p - x1);
    if (cross <= 0):  # P X1 X2 is obtuse angle
        return np.linalg.norm(p - x1), 0.0

    d2 = np.dot(x2 - x1, x2 - x1)
    if (cross >= d2):  # P X2 X1 is obtuse angle
        return np.linalg.norm(p - x2), 1.0

    alpha = cross / d2
    q = x1 + alpha * (x2 - x1)  # q is the foot of the perpendicular from p
    return np.linalg.norm(p - q), alpha



def _line_intersect(x1, x2, y1, y2):
    """
    compute the intersection point of line x1,x2 and line y1,y2, the intersection point is (1-alpha)*x1 + alpha*x2 = (1-beta)*y1 + beta*y2

    Args:
        x1: 2d numpy vector, first node of segment one
        x2: 2d numpy vector, second node of segment one
        y1: 2d numpy vector, first node of segment two
        y2: 2d numpy vector, second node of segment two

    Returns:
        float alpha and float beta

    """

    A = [[x2[0] - x1[0], y1[0] - y2[0]], [x2[1] - x1[1], y1[1] - y2[1]]]
    b = [y1[0]] - x1[0], y1[1] - x1[1]
    delta = A[0][0] * A[1][1] - A[0][1] * A[1][0]

    if A[0][0] ** 2 + A[1][0] ** 2 < 1e-16:
        print(x1, x2, y1, y2)

    if (np.fabs(delta) < 1e-16):
        if (np.fabs(np.cross(x2 - x1, y1 - x1)) > 1e-16):
            return np.inf, np.inf
        else:
            # two line overlaps

            alpha_1, alpha_2 = [np.dot(y1 - x1, x2 - x1), np.dot(y2 - x1, x2 - x1)] / np.dot(x2 - x1, x2 - x1)
            beta_1, beta_2 = [np.dot(x1 - y1, y2 - y1), np.dot(x2 - y1, y2 - y1)] / np.dot(y2 - y1, y2 - y1)
            if (alpha_1 * alpha_2 <= 1e-16):
                alpha = 0
            else:
                alpha = np.minimum(alpha_1, alpha_2)
            if (beta_1 * beta_2 <= 1e-16):
                beta = 0
            else:
                beta = np.minimum(beta_1, beta_2)
            return alpha, beta

    return (A[1][1]*b[0] - A[0][1]*b[1])/delta, (-A[1][0]*b[0] + A[0][0]*b[1])/delta


class Intersector_Base:
    def __init__(self, fluid, structure):
        """
        construct intersector

        Args:
            fluid: Fluid_Domain class
            structure: Structure class

        Attributes:
            bounding_boxes: float[number of fluid vertices, 4], a list of bounding boxes, for fluid nodes.
                            Each box contains the fluid node and its neighbors, as x_min, y_min, x_max,y_max

            candidates: list of structure edge list like [[],[],[]].  if node i is close to IB, candidates[i] is not empty,
                        stores all structure edge id, structure edge j's bounding box intersects node i' bounding box

            neighbors: reference, list of list fluid vertex neighbors, neighbors[i] contains node i 's neighbor nodes

            connectivity: list of fluid vertex list, connectivity[i] contains i's neighbor, which is in the save fluid side,
                          the edge is not intersected.

            edge_center_stencil : (int,int,float)[number of fluid edges], for those ghost edge center which has one ghost node,
                                  its stencil as dante's definition and its intersection on the stencil

            edge_center_closest_position : (int float)[nedges] , for those ghost edge center which has one ghost node, its closest
                                          structure edge and its position

            ghost_node_closest_position : (int float)[nedges] , for those ghost nodes near IB, its closest structure edge and its position

            ghost_node_stencil : (int, int,float,int,int, float)[nedges], for those ghost nodes near IB, saying xb its stencils node numbers
                                 as dante's definition and its intersection on the stencil, for both side.  The first 3 number is for the first stencil,
                                 which is on xs -> x_b side(the same side as x_b), here xs is the closest point at structure;  the second three number is for the second
                                 stencil, which is on  x_b -> xs side(the opposite side of x_b).

            is_close: bool[number of fluid vertices], true if vert is close to immersed boundary(its bounding box intersects with some structure edge bounding box)

            intersect_or_not: bool[number of fluid edges], true if the fluid edge intersects with structure

            intersect_result: (float, int, float, float, int , float)[number of fluid edges],
                              alpha_1 ,s1, beta_1 ,alpha_2 ,s2, beta_2
                              alpha, save the intersection point information for fluid edges, saying fluid edge (x1, x2)
                              alpha1 is for intersecting point from left to right, which is  x1 + alpha1*(x2 - x1), which is also the point (s1, beta1) on structure
                              alpha1 is for intersecting point from right to left, which is  x2 + alpha2*(x1 - x2), which is also the point (s2, beta2) on structure




            nedges:int, fluid edge number

            edges: reference, float[number of fluid edges, 2] ,  fluid edge nodes

            nverts:int, fluid vertex number

            struc_box: float[4], the bounding box of structure

            struc_nedges:int, structure boundary edge number

            struc_nverts:int, structure boundary vertex number

            struc_verts:reference, structure vertex coordinate

            struc_edges:reference, structure edges

            status_in_fluid: bool[number of fluid vertices], true if the vertex is in the fluid domain

            status : bool[number of fluid vertices], true if the vertex's whole control volume is in the fluid domain

            verts: edges: reference, float[number of fluid vertices, 2] ,  fluid vertex coordinates

        Returns:
            None
            _build_fluid_bounding_boxes()
            _build_close_node_result()
            _build_intersect_result()
            _initial_status()
            _compute_HO_stencil()
            _compute_ghost_stencil()
        """
        self.edges = fluid.edges
        self.nedges = fluid.nedges
        self.elems = fluid.elems

        self.verts = fluid.verts
        self.nverts = fluid.nverts

        self.neighbors = fluid.neighbors
        self.node_elem_neighbors = fluid.node_elem_neighbors

        self.bounding_boxes = np.empty(shape=[self.nverts, 4], dtype=float)
        self.struc_box = np.array([np.inf, np.inf, -np.inf, -np.inf], dtype=float)

        self.struc_nedges = structure.nbounds
        self.struc_nverts = structure.nverts
        self.struc_verts = structure.verts
        self.struc_edges = structure.bounds


        self.is_close = np.empty(shape=[self.nverts], dtype=bool)
        self.status_in_fluid = np.empty(shape=[self.nverts], dtype=bool)
        self.status = np.empty(shape=[self.nverts], dtype=bool)

        self.intersect_or_not = np.empty(shape=[self.nedges], dtype=bool)
        self.intersect_result = np.empty(shape=[self.nedges], dtype='float,int,float,float,int,float')


        self.edge_center_stencil = np.empty(shape=[self.nedges], dtype='int,int,float')

        self.edge_center_closest_position = np.empty(shape=[self.nedges], dtype='int, float')

        self.ghost_node_stencil = np.empty(shape=[self.nverts], dtype='int,int,float,int,int,float')

        self.ghost_node_closest_position = np.empty(shape=[self.nedges], dtype='int, float')

        self._build_fluid_bounding_boxes()
        self._build_close_node_result()
        self._build_intersect_result()
        self._build_connectivity()
        self._initial_status_in_fluid()
        self._initial_status()
        #self._compute_HO_stencil()
        #self._compute_ghost_stencil()

    def _build_fluid_bounding_boxes(self):
        """
        build fluid bounding boxes(self.bounding_boxes) and structure bounding box(self.struc_box)

        Args:

        Returns:


        """
        print('start building bounding box')
        verts = self.verts
        bounding_boxes = self.bounding_boxes
        for i in range(self.nverts):
            box = bounding_boxes[i, :]
            box[0:2] = box[2:4] = verts[i, :]
            for j in self.neighbors[i]:
                x, y = verts[j, :]
                if (x < box[0]):
                    box[0] = x
                if (x > box[2]):
                    box[2] = x
                if (y < box[1]):
                    box[1] = y
                if (y > box[3]):
                    box[3] = y

        struc_box = self.struc_box
        for x, y in self.struc_verts:
            if (x < struc_box[0]):
                struc_box[0] = x
            if (x > struc_box[2]):
                struc_box[2] = x
            if (y < struc_box[1]):
                struc_box[1] = y
            if (y > struc_box[3]):
                struc_box[3] = y
        print('finish building bounding box')

    def _build_close_node_result(self):
        """
        find out all nodes closed to IB, and label them in is_close; construct candidates list for them, containing
        its close structure edge id

        Args:
            None
        Returns:
            None
        """
        print('start building close node result')
        bounding_boxes = self.bounding_boxes
        struc_box = self.struc_box
        is_close = self.is_close
        for n in range(self.nverts):
            is_close[n] = _box_intersect(bounding_boxes[n, :], struc_box)

        struc_edges = self.struc_edges
        struc_verts = self.struc_verts
        struc_nedges = self.struc_nedges
        struc_edge_bounding_boxes = np.empty(shape=[struc_nedges, 4], dtype=float)
        for i in range(struc_nedges):
            n1, n2 = struc_edges[i]
            x1, y1 = struc_verts[n1]
            x2, y2 = struc_verts[n2]
            struc_edge_bounding_boxes[i, :] = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


        self.candidates = candidates = [[] for i in range(self.nverts)]
        for n in range(self.nverts):
            if is_close[n]:
                is_close[n] = _box_candidate(bounding_boxes[n, :], struc_edge_bounding_boxes, candidates[n])

        print('finish building close node result')

    def _build_intersect_result(self):
        """
        find all fluid edge, intersecting structure embedded surface, label them in intersect_or_not,
        and save the intersection points for both side in intersect_result

        Args:
            None
        Returns:
            None

        """
        print('start building intersect result')

        is_close = self.is_close
        intersect_or_not = self.intersect_or_not
        intersect_result = self.intersect_result
        intersect_or_not[:] = False

        for i in range(self.nedges):
            n1, n2 = self.edges[i, :]
            if (is_close[n1] and is_close[n2]):

                alpha, s_i, s_beta = self._build_intersect_result_helper(n1, n2)
                if (alpha != np.inf):
                    intersect_or_not[i] = True
                    intersect_result[i][0], intersect_result[i][1],intersect_result[i][2] = alpha, s_i, s_beta

                alpha, s_i, s_beta = self._build_intersect_result_helper(n2, n1)
                if (alpha != np.inf):
                    intersect_or_not[i] = True
                    intersect_result[i][3], intersect_result[i][4], intersect_result[i][5] = alpha, s_i, s_beta
        print('finish building intersect result')

    def _build_intersect_result_helper(self, n1, n2):
        """
        compute fluid edge intersection information, fluid edge nodes n1 and n2

        :param n1: fluid node number
        :param n2: fluid node number
        :return: alpha,the intersection point is x1 + alpha*(x2-x1)
        """
        struc_verts = self.struc_verts
        min_alpha , min_i, min_beta = np.inf, -1, 0.0
        cands = []
        for i in self.candidates[n1]:
            for j in self.candidates[n2]:
                if i == j:
                    cands.append(i)

        x1,x2 = self.verts[n1],self.verts[n2]


        for i in cands:
            s_n1, s_n2 = self.struc_edges[i]
            y1, y2 = struc_verts[s_n1, :], struc_verts[s_n2, :]
            alpha, beta = _line_intersect(x1, x2, y1, y2)
            if (0 <= alpha <= 1 and 0 <= beta <= 1 and alpha < min_alpha):
                min_alpha = alpha
                min_beta = beta
                min_i = i
        return min_alpha, min_i, min_beta



    def _build_connectivity(self):

        # build connect set
        intersect_or_not = self.intersect_or_not
        self.connectivity = connectivity = [[] for i in range(self.nverts)]
        for i in range(self.nedges):
            n1, n2 = self.edges[i, :]
            if (not intersect_or_not[i]):
                connectivity[n1].append(n2)
                connectivity[n2].append(n1)

    def _initial_status_in_fluid(self):
        """
        Initialize node in fluid state by flood filling method, assume the nodes at top are active

        Args:
            None
        Returns:
            None

        """
        print('start initializing status in fluid')
        status_in_fluid = self.status_in_fluid
        status_in_fluid[:] = False

        connectivity = self.connectivity



        y_max = -np.inf
        for n in range(self.nverts):
            y = self.verts[n, 1]
            if (y > y_max):
                active_node = n
                y_max = y
        status_in_fluid[active_node] = True

        flood_fill = [active_node]
        while flood_fill:
            current_node = flood_fill.pop()
            for neighbor in connectivity[current_node]:
                if (not status_in_fluid[neighbor]):
                    status_in_fluid[neighbor] = True
                    flood_fill.append(neighbor)

        print('finish initializing status in fluid')


    def _initial_status(self):
        print('start initializing status')
        intersect_or_not = self.intersect_or_not
        intersect_result = self.intersect_result
        self.status = status = copy.copy(self.status_in_fluid)
        for i in range(self.nedges):
            if (intersect_or_not[i]):
                n1, n2 = self.edges[i, :]
                alpha_1, alpha_2 = intersect_result[i, 0],intersect_result[i, 3]
                if (alpha_1 <= 0.5):
                    status[n1] = False
                if (alpha_2 <= 0.5):
                    status[n2] = False
        print('finish initializing status')

    def _compute_edge_center_closest_position(self):
        status = self.status
        verts = self.verts
        edges = self.edges
        for i in range(self.nedges):
            n1, n2 = edges[i, :]
            if ((status[n1] and not status[n2]) or (status[n2] and not status[n1])):
                x_c = 0.5 * (verts[n1, :] + verts[n2, :])
                dist, struc_edge_id, alpha = self._compute_closest_position_helper(x_c)
                self.edge_center_closest_position[i] = struc_edge_id, alpha

    def _compute_closest_position_helper(self, x):
        struc_edges = self.struc_edges
        struc_verts = self.struc_verts
        min_dist = np.inf

        for i in range(self.struc_nedges):
            n1, n2 = struc_edges[i, :]
            x1 = struc_verts[n1, :]
            x2 = struc_verts[n2, :]
            dist, alpha = _compute_dist(x, x1, x2)
            if (dist < min_dist):
                min_dist = dist
                argmin_edge_id = i
                min_alpha = alpha

        return min_dist, argmin_edge_id, min_alpha

    '''
    def _compute_HO_stencil_old(self):
        status = self.status
        verts = self.verts
        edges = self.edges
        struc_edges = self.struc_edges
        struc_verts = self.struc_verts

        self._compute_edge_center_closest_position()

        edge_center_stencil = self.edge_center_stencil
        edge_center_closest_position = self.edge_center_closest_position
        neighbors = self.neighbors

        for i in range(self.nedges):
            n1,n2 = edges[i,:]
            if((status[n1] and not status[n2]) or (status[n2] and not status[n1])):

                x_c = 0.5*(verts[n1,:] + verts[n2,:])
                struc_edge_id,struc_alpha = edge_center_closest_position[i]

                n_s1,n_s2 = struc_edges[struc_edge_id]
                x_b = (1 - struc_alpha)*struc_verts[n_s1] + struc_alpha*struc_verts[n_s2]

                min_alpha = np.inf
                stencil_status = False


                for n in [n1,n2]:
                    x1 = verts[n]
                    for m in neighbors[n]:
                        if(not status[n] and not status[m]): # ignore stencil with 2 inactive nodes
                            continue
                        x2 = verts[m]

                        alpha,beta = _line_intersect(x_b,x_c,x1,x2)


                        if(0 <= beta <= 1.0): # intersect the edge
                            good_stencil = status[n] and status[m]


                            if(good_stencil): # stencil has 2 active nodes
                                if (not stencil_status or alpha < min_alpha ):
                                    min_alpha = alpha
                                    min_beta = beta
                                    n_p,n_q = n,m
                                    stencil_status = True

                            elif(not stencil_status and alpha < min_alpha):# stencil has 1 active node

                                min_alpha = alpha
                                min_beta = beta
                                n_p,n_q = n,m


                if(min_alpha == np.inf):
                    print("error in compute HO stencil")
                edge_center_stencil[i] = n_p,n_q,min_beta

                #x_s = x_b + min_alpha*(x_c - x_b)
                #plt.plot([x_b[0],x_s[0]],[x_b[1],x_s[1]],color = 'y')
        #plt.show()
        '''

    def _compute_HO_stencil(self):
        print('starting computing HO stencil')
        status = self.status
        verts = self.verts
        edges = self.edges

        self._compute_edge_center_closest_position()

        edge_center_stencil = self.edge_center_stencil
        edge_center_closest_position = self.edge_center_closest_position

        for i in range(self.nedges):
            n1, n2 = edges[i, :]
            if ((status[n1] and not status[n2]) or (status[n2] and not status[n1])):
                x_c = 0.5 * (verts[n1, :] + verts[n2, :])

                stencil, min_beta = self._compute_stencil(x_c, edge_center_closest_position[i], [n1, n2])

                edge_center_stencil[i] = stencil[0][0], stencil[0][1], min_beta[0]

                # x_s = x_b + min_alpha*(x_c - x_b)
                # plt.plot([x_b[0],x_s[0]],[x_b[1],x_s[1]],color = 'y')
        # plt.show()
        print('finish computing HO stencil')

    def _compute_stencil(self, x_b, xs_info, nodes, num_side=1):
        '''

        :param nodes: candidate nodes
        :param xs,ys: its closest points on IB
        :return:

        check all element nodes of neighbor nodes for candidates
        '''

        s_i, s_alpha = xs_info

        s_n, s_m = self.struc_edges[s_i]

        s_xl, s_xr = self.struc_verts[s_n], self.struc_verts[s_m]

        x_s = (1 - s_alpha) * s_xl + s_alpha * s_xr


        min_alpha = [np.inf, np.inf]

        min_beta = [0.0] * num_side

        stencil = [[-1, -1]] * num_side

        stencil_status = [False, False]

        neighbors = self.neighbors
        node_elem_neighbors = self.node_elem_neighbors
        status = self.status
        verts = self.verts
        elems = self.elems

        for n in nodes:


            stencil_status[:] = False, False

            for neigh in neighbors[n]:
                for ele in node_elem_neighbors[neigh]:
                    ele_node = elems[ele]
                    for i in range(-1, 2):
                        n1, n2 = ele_node[i], ele_node[i + 1]

                        if (not status[n1] and not status[n2]):  # ignore stencil with 2 inactive nodes
                            continue
                        x1, x2 = verts[n1], verts[n2]

                        alpha, beta = _line_intersect(x_s, x_b, x1, x2)

                        if (0 <= beta <= 1.0):  # intersect the edge
                            good_stencil = status[n1] and status[n2]

                            side = 0 if alpha >= -1e-14 else 1

                            if (num_side == 1 and side == 1):
                                continue

                            alpha = np.fabs(alpha)

                            if (good_stencil):  # stencil has 2 active nodes
                                if (not stencil_status[side] or alpha < min_alpha[side]):
                                    min_alpha[side] = alpha
                                    min_beta[side] = beta
                                    stencil[side] = n1, n2
                                    stencil_status[side] = True

                            elif (not stencil_status[side] and alpha < min_alpha[side]):  # stencil has 1 active node

                                min_alpha[side] = alpha
                                min_beta[side] = beta
                                stencil[side] = n1, n2

        return stencil, min_beta

    def _compute_ghost_stencil(self):
        print('starting computing ghost stencil')
        status = self.status
        is_close = self.is_close
        nverts = self.nverts
        verts = self.verts

        ghost_node_closest_position = self.ghost_node_closest_position

        ghost_node_stencil = self.ghost_node_stencil

        for n in range(nverts):
            ghost_node_stencil[n] = -1,-1,0.0,-1,-1,0.0

            if (not is_close[n] or status[n]):  # skip these active or far from IB nodes
                continue
            p = verts[n]
            dist, struc_edge_id, alpha = self._compute_closest_position_helper(p)
            ghost_node_closest_position[n] = struc_edge_id, alpha

            stencil, min_beta = self._compute_stencil(p, ghost_node_closest_position[n], [n], num_side=2)

            ghost_node_stencil[n] = stencil[0][0], stencil[0][1], min_beta[0], stencil[1][0], stencil[1][1], min_beta[1]
        print('finish computing ghost stencil')

    def draw(self, node_status=False, node_number=False, edge_center_stencil=False, ghost_stencil=False,
             close_points=False):
        edges = self.edges
        nedges = self.nedges

        verts = self.verts
        nverts = self.nverts

        struc_nedges = self.struc_nedges
        struc_verts = self.struc_verts
        struc_edges = self.struc_edges

        status = self.status
        intersect = self.intersect_or_not
        is_close = self.is_close

        # intersected edge red; not intersected edge blue
        for i in range(nedges):
            n1, n2 = edges[i, :]
            x1, x2 = verts[n1], verts[n2]

            if (intersect[i]):
                plt.plot([x1[0], x2[0]], [x1[1], x2[1]], color='r')
            else:
                plt.plot([x1[0], x2[0]], [x1[1], x2[1]], color='b')

        # draw edge center stencil

        if (close_points):
            for n in range(nverts):
                if is_close[n]:
                    x, y = verts[n]
                    plt.text(x, y, '%d' % n, color='blue')

        if (edge_center_stencil):
            for i in range(nedges):
                n1, n2 = edges[i, :]

                if ((status[n1] and not status[n2]) or (status[n2] and not status[n1])):
                    n_p, n_q, alpha = self.edge_center_stencil[i]

                    x_p, x_q = verts[n_p], verts[n_q]

                    x_1, x_2 = verts[n1], verts[n2]

                    x_stencil = (1 - alpha) * x_p + alpha * x_q

                    x_c = (x_1 + x_2) / 2

                    s_i, s_alpha = self.edge_center_closest_position[i]

                    x_s1, x_s2 = struc_verts[struc_edges[s_i][0]], struc_verts[struc_edges[s_i][1]]

                    x_s = (1 - s_alpha) * x_s1 + s_alpha * x_s2

                    plt.plot([x_s[0], x_c[0], x_stencil[0]], [x_s[1], x_c[1], x_stencil[1]], color='k')

                    plt.plot(x_s[0], x_s[1], 'ok')

                    plt.plot(x_c[0], x_c[1], 'ok')

                    plt.plot(x_stencil[0], x_stencil[1], 'ok')

        # draw ghost node stencil
        if (ghost_stencil):
            for n in range(nverts):

                if (not is_close[n] or status[n]):  # skip these active or far from IB nodes
                    continue

                x_n = verts[n]

                s_n, s_alpha = self.ghost_node_closest_position[n]

                x_s1, x_s2 = struc_verts[struc_edges[s_n][0]], struc_verts[struc_edges[s_n][1]]

                x_s = (1 - s_alpha) * x_s1 + s_alpha * x_s2

                plt.plot(x_s[0], x_s[1], 'ok')

                plt.plot(x_n[0], x_n[1], 'ok')


                for i in range(2):
                    n_p, n_q, alpha = self.ghost_node_stencil[n][3 * i], self.ghost_node_stencil[n][3 * i + 1], \
                                      self.ghost_node_stencil[n][3 * i + 2]

                    if (n_p == -1 and n_q == -1):
                        continue

                    x_p, x_q = verts[n_p], verts[n_q]



                    x_stencil = (1 - alpha) * x_p + alpha * x_q

                    plt.plot([x_s[0], x_n[0], x_stencil[0]], [x_s[1], x_n[1], x_stencil[1]], color='k')


                    if (i == 0):
                        plt.plot(x_stencil[0], x_stencil[1], 'ok')
                    else:
                        plt.plot(x_stencil[0], x_stencil[1], 'or')

        # structure green
        for i in range(struc_nedges):
            n1, n2 = struc_edges[i, :]
            x1, x2 = struc_verts[n1], struc_verts[n2]
            plt.plot([x1[0], x2[0]], [x1[1], x2[1]], color='g')

        if (node_number):
            # active vertex blue; inactive vertex red
            for n in range(nverts):
                x, y = verts[n]
                plt.text(x, y, '%d' % n, color = 'blue')

        if (node_status):

            # active vertex blue; inactive vertex red
            for n in range(nverts):
                x, y = verts[n]
                if (status[n]):

                    plt.text(x, y, '%d' % n, color='green')

                else:
                    plt.text(x, y, '%d' % n, color='red')




        plt.axis([-2, 2, -2, 2])

        plt.show()


if __name__ == "__main__":
    fluid = Fluid_Domain('../Test/IBNaca/domain')
    structure = Structure('../Test/IBNaca/naca')
    intersector = Intersector(fluid, structure)
    # intersector.draw(True,  False)

    #draw(self, node_status=False, node_number=False, edge_center_stencil=False, ghost_stencil=False,close_points=False)

    intersector.draw(False, False, False, True,True)
    #intersector.draw(False, False,  False, False, False)
