import numpy as np
import matplotlib.pyplot as plt
from Fluid_Domain import Fluid_Domain
from Structure import Structure

def _box_intersect(box1,box2):
    # lazy intersection
    # mincorner: box[0], box[1]
    # maxcorner: box[2], box[3]
    return (box1[0] <= box2[2] and box1[1] <= box2[3] and box2[0] <= box1[2] and box2[1] <= box1[3])

def _compute_dist(p,x1,x2):
    # compute the distance between vertex p, and segment (x1,x2)
    # return distance,alpha
    # the intersection point is (1-alpha) * x1 + alpha*x2

    cross = np.dot(x2 - x1, p - x1);
    if (cross <= 0): # P X1 X2 is obtuse angle
        return np.linalg.norm(p-x1), 0.0

    d2 = np.dot(x2-x1,x2-x1)
    if (cross >= d2):  # P X2 X1 is obtuse angle
        return np.linalg.norm(p - x2), 1.0

    alpha = cross/d2
    q = x1 + alpha*(x2 -x1) # q is the foot of the perpendicular from p
    return np.linalg.norm(p - q), alpha

def _line_intersect(x1,x2,y1,y2):
    # compute the intersection of line x1,x2 and line y1,y2
    # x1 + alpha*(x2-x1) = y1 + beta*(y2-y1)
    # return intersect, alpha, beta
    delta = np.cross(x2-x1,y2-y1)
    A = np.array([[x2[0] - x1[0],y1[0] - y2[0]], [x2[1] - x1[1], y1[1] - y2[1]]])
    b = y1 - x1

    if(np.fabs(delta) < 1e-16):
        if(np.fabs(np.cross(x2-x1,y1-x1)) > 1e-16):
            return np.inf,np.inf
        else:
            # two line overlaps

            alpha_1,alpha_2 = [np.dot(y1-x1, x2-x1), np.dot(y2-x1, x2-x1)] / np.dot(x2-x1, x2-x1)
            beta_1, beta_2  = [np.dot(x1-y1, y2-y1) , np.dot(x2-y1, y2-y1)]  / np.dot(y2-y1, y2-y1)
            if(alpha_1 * alpha_2 <= 1e-16):
                alpha = 0
            else:
                alpha = np.minimum(alpha_1,alpha_2)
            if(beta_1 * beta_2 <= 1e-16):
                beta = 0
            else:
                beta = np.minimum(beta_1,beta_2)
            return alpha, beta


    return np.linalg.solve(A, b)


class Intersector:
    def __init__(self,fluid,structure):
        ########################################################################################
        #######       NAME TABLE
        ########################################################################################
        # nedges:int, fluid edge number
        # nedges:ref, fluid edges
        # verts: ref, fluid verts coordinate
        # nverts:int, fluid vert number
        # connectivity: ref, fluid vert connectivity
        # bounding_boxes: float[nverts,4], bounding box of vert and its neighbors, as x_min, y_min, x_max,y_max
        # struc_nedges:int, structure boundary edge number
        # struc_nverts:int, structure boundary vert number
        # struc_verts:ref, structure verts
        # struc_edges,ref, structure edges
        # is_close, 
        self.edges = fluid.edges
        self.nedges = fluid.nedges

        self.verts = fluid.verts
        self.nverts = fluid.nverts

        self.connectivity = fluid.connectivity

        self.bounding_boxes = np.empty(shape=[self.nverts,4],dtype=float)
        self.struc_box  = np.array([np.inf,np.inf,-np.inf,-np.inf],dtype=float)

        self.struc_nedges = structure.nbounds
        self.struc_nverts = structure.nverts
        self.struc_verts = structure.verts
        self.struc_edges = structure.bounds

        self.is_close = np.empty(shape=[self.nverts],dtype=bool)
        self.status_in_fluid = np.empty(shape=[self.nverts],dtype=bool)
        self.status = np.empty(shape=[self.nverts],dtype=bool)

        self.intersect_or_not = np.empty(shape=[self.nedges],dtype=bool)
        self.intersect_result = np.empty(shape=[self.nedges,2],dtype=float)

        self.edge_center_stencil = np.empty(shape=[self.nedges],dtype='int,int,float')

        self.edge_center_closest_position = np.empty(shape=[self.nedges],dtype='int, float')

        self._build_fluid_bounding_boxes()
        self._build_intersect_result()
        self._initial_status()
        self._compute_HO_stencil()


    def _build_fluid_bounding_boxes(self):
        verts = self.verts
        bounding_boxes = self.bounding_boxes
        for i in range(self.nverts):
            box = bounding_boxes[i,:]
            box[0:2] = box[2:4] = verts[i,:]
            for j in self.connectivity[i]:
                x,y = verts[j,:]
                if(x < box[0]):
                    box[0] = x
                if(x > box[2]):
                    box[2] = x
                if(y < box[1]):
                    box[1] = y
                if(y > box[3]):
                    box[3] = y

        struc_box = self.struc_box
        for x,y in self.struc_verts:
            if(x < struc_box[0]):
                struc_box[0] = x
            if(x > struc_box[2]):
                struc_box[2] = x
            if(y < struc_box[1]):
                struc_box[1] = y
            if(y > struc_box[3]):
                struc_box[3] = y

    def _build_intersect_result(self):
        bounding_boxes = self.bounding_boxes
        struc_box = self.struc_box
        is_close = self.is_close
        intersect_or_not = self.intersect_or_not
        intersect_result = self.intersect_result
        intersect_or_not[:] = False
        for n in range(self.nverts):
            is_close[n] = _box_intersect(bounding_boxes[n,:],struc_box)


        verts = self.verts
        for i in range(self.nedges):
            n1,n2 = self.edges[i,:]
            if(is_close[n1] and is_close[n1]):
                x1,x2 = verts[n1,:],verts[n2,:]
                alpha = self._build_intersect_result_helper(x1,x2)
                if(alpha != np.inf):
                    intersect_or_not[i] = True
                intersect_result[i,0] = alpha

                alpha = self._build_intersect_result_helper(x2,x1)
                if(alpha != np.inf):
                    intersect_or_not[i] = True
                intersect_result[i,1] = alpha

    def _build_intersect_result_helper(self,x1,x2):
        struc_verts = self.struc_verts
        min_alpha = np.inf
        for n1, n2 in self.struc_edges:
            y1,y2 = struc_verts[n1,:],struc_verts[n2,:]
            alpha,beta = _line_intersect(x1,x2,y1,y2)
            if(0 <= alpha <= 1 and 0 <= beta <= 1 and alpha < min_alpha):
                min_alpha = alpha
        return min_alpha



    #def _initial_status_in_fluid(self):

    def _initial_status(self):
        status = self.status
        status[:] = False
        is_close = self.is_close
        intersect_or_not = self.intersect_or_not
        intersect_result = self.intersect_result
        #flood fill method to initialize, assume the nodes at top are active
        #build connect set
        connect = [[] for i in range(self.nverts)]
        for i in range(self.nedges):
            n1,n2 = self.edges[i,:]
            if(not intersect_or_not[i]):
                connect[n1].append(n2)
                connect[n2].append(n1)

        y_max = -np.inf
        for n in range(self.nverts):
            y = self.verts[n,1]
            if(y > y_max):
                active_node = n
                y_max = y
        status[active_node] = True

        flood_fill = [active_node]
        while flood_fill:
            current_node = flood_fill.pop()
            for neighbor in connect[current_node]:
                if(not status[neighbor]):
                    status[neighbor] = status[current_node]
                    flood_fill.append(neighbor)



        for i in range(self.nedges):
            if(intersect_or_not[i]):
                n1,n2 = self.edges[i,:]
                alpha_1, alpha_2 = intersect_result[i,:]
                if(alpha_1 <= 0.5):
                    status[n1] = False
                if(alpha_2 <= 0.5):
                    status[n2] = False


    def _compute_closest_position(self):
        status = self.status
        verts = self.verts
        edges = self.edges
        for i in range(self.nedges):
            n1,n2 = edges[i,:]
            if((status[n1] and not status[n2]) or (status[n2] and not status[n1])):
                x_c = 0.5*(verts[n1,:] + verts[n2,:])
                dist, struc_edge_id, alpha = self._compute_closest_position_helper(x_c)
                self.edge_center_closest_position[i] = struc_edge_id,alpha

    def _compute_closest_position_helper(self,x):
        struc_edges = self.struc_edges
        struc_verts = self.struc_verts
        min_dist = np.inf

        for i in range(self.struc_nedges):
            n1,n2 = struc_edges[i,:]
            x1 = struc_verts[n1,:]
            x2 = struc_verts[n2,:]
            dist,alpha = _compute_dist(x,x1,x2)
            if(dist < min_dist):
                min_dist = dist
                argmin_edge_id = i
                min_alpha = alpha

        return min_dist, argmin_edge_id, min_alpha



    def _compute_HO_stencil(self):
        status = self.status
        verts = self.verts
        edges = self.edges
        struc_edges = self.struc_edges
        struc_verts = self.struc_verts

        self._compute_closest_position()

        edge_center_stencil = self.edge_center_stencil
        edge_center_closest_position = self.edge_center_closest_position
        connectivity = self.connectivity

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
                    for m in connectivity[n]:
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

                            elif(not stencil_status and min_alpha < alpha):# stencil has 1 active node

                                min_alpha = alpha
                                min_beta = beta
                                n_p,n_q = n,m


                if(min_alpha == np.inf):
                    print("error in compute HO stencil")
                edge_center_stencil[i] = n_p,n_q,min_beta

                #x_s = x_b + min_alpha*(x_c - x_b)
                #plt.plot([x_b[0],x_s[0]],[x_b[1],x_s[1]],color = 'y')
        #plt.show()





    def draw(self):
        edges = self.edges
        nedges = self.nedges

        verts = self.verts
        nverts = self.nverts

        struc_nedges = self.struc_nedges
        struc_nverts = self.struc_nverts
        struc_verts = self.struc_verts
        struc_edges = self.struc_edges

        status = self.status
        intersect = self.intersect_or_not
        #intersected edge red; not intersected edge blue
        for i in range(nedges):
            n1,n2 = edges[i,:]
            x1,x2 = verts[n1],verts[n2]

            if(intersect[i]):
                plt.plot([x1[0],x2[0]],[x1[1],x2[1]],color = 'r')
            else:
                plt.plot([x1[0],x2[0]],[x1[1],x2[1]], color = 'b')
        '''
        for i in range(nedges):
            n1,n2 = edges[i,:]

            if((status[n1] and not status[n2]) or (status[n2] and not status[n1])):
                n_p,n_q = self.edge_center_stencil[i,:]
                x_p,x_q = verts[n_p],verts[n_q]
                plt.plot([x_p[0],x_q[0]],[x_p[1],x_q[1]],color = 'k')
        '''

        #structure greee
        for i in range(struc_nedges):
            n1,n2 = struc_edges[i,:]
            x1,x2 = struc_verts[n1],struc_verts[n2]
            plt.plot([x1[0],x2[0]],[x1[1],x2[1]],color = 'g')
        #active vertex blue; inactive vertex red
        for n in range(nverts):
            x = verts[n]
            if(status[n]):
                plt.plot(x[0],x[1], 'ob')
            else:
                plt.plot(x[0],x[1], 'or')
        plt.axis([-2, 2, -2, 2])
        plt.show()






'''
fluid = Fluid_Domain('blasius_coarse.fgrid')
structure = Structure('plate.fgrid')
intersector = Intersector(fluid,structure)
intersector.draw()
'''