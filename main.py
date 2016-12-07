#  ##!/usr/bin/env python
#from Embedded_Explicit_Solver_New import *
from Embedded_Explicit_Solver_FIVER2 import *
from Structure import Structure
from Io_Data import *
'''
def main():
    # get input parameter
    # read fluid grid & construct fluid grid data

    io_data = Io_Data("../Test/NacaBF/"
                      "Invisc/naca.input")
    #io_data = Io_Data("../Test/Wedge/wedge.input")
    #io_data = Io_Data("../Test/EulerNaca/naca.input")
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
'''
def main():
    # get input parameter
    # read fluid grid & construct fluid grid data


    io_data = Io_Data("../Test/IBNaca/naca.input")
    fluid_input = io_data.fluidmesh
    structure_input = io_data.structuremesh

    fluid = Fluid_Domain(fluid_input)

    structure = Structure(structure_input)
    #fluid.check_mesh(True)



    explicit_solver = Embedded_Explicit_Solver(fluid,structure,io_data)

    explicit_solver._solve()
    explicit_solver._draw_solution()
    np.save("nacaW1",explicit_solver.W)

    # if it is immersed boundary problem



    # read body grid & construct body grid data

    # construct intersector

    # set_initial_solution

main()