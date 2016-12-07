from Intersector_Base import *

class Intersector(Intersector_Base):
    def __init__(self, fluid, structure):
        super().__init__(fluid, structure)
        self._compute_HO_stencil()
        self._compute_ghost_stencil()

    def _initial_status(self):
        print('start initializing status')
        intersect_or_not = self.intersect_or_not
        intersect_result = self.intersect_result
        self.status = status = copy.copy(self.status_in_fluid)
        for i in range(self.nedges):
            if (intersect_or_not[i]):
                n1, n2 = self.edges[i, :]
                alpha_1, alpha_2 = intersect_result[i][0],intersect_result[i][3]
                if (alpha_1 <= 0.5):
                    status[n1] = False
                if (alpha_2 <= 0.5):
                    status[n2] = False
        print('finish initializing status')