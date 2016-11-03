#  ##!/usr/bin/env python
from Embedded_Explicit_Solver import *
from Io_Data import *

def main():
    # get input parameter
    # read fluid grid & construct fluid grid data

    #io_data = Io_Data("../Test/Blasius/blasius.input")
    io_data = Io_Data("../Test/Wedge/wedge.input")
    fluid_input = io_data.fluidmesh

    fluid = Fluid_Domain(fluid_input)


    explicit_solver = Explicit_Solver(fluid,io_data)

    explicit_solver._solve()
    explicit_solver._draw_solution()
    np.save("solutionW",explicit_solver.W)

    # if it is immersed boundary problem



    # read body grid & construct body grid data

    # construct intersector

    # set_initial_solution

main()