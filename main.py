#  ##!/usr/bin/env python
from Explicit_Solver import *
from Io_Data import *

def main():
    # get input parameter
    # read fluid grid & construct fluid grid data

    io_data = Io_Data("../Test/NacaBF/Visc/naca.input")
    #io_data = Io_Data("../Test/Wedge/wedge.input")
    fluid_input = io_data.fluidmesh

    fluid = Fluid_Domain(fluid_input)
    #fluid.check_mesh(True)



    explicit_solver = Explicit_Solver(fluid,io_data)

    explicit_solver._solve()
    explicit_solver._draw_solution()
    #np.save("nacaW",explicit_solver.W)

    # if it is immersed boundary problem



    # read body grid & construct body grid data

    # construct intersector

    # set_initial_solution

main()