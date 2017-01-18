from Intersector_Base import *

class Intersector(Intersector_Base):
    def __init__(self, fluid, structure):
        super().__init__(fluid, structure)


    def _initial_status(self):
        print('start initializing status')
        self.status  = copy.copy(self.status_in_fluid)
        print('finish initializing status')