#  ##!/usr/bin/env python
from Io_Data import *


'''
#this is for bodyfitted mesh simulation
from Explicit_Solver import *
def main():
    # get input parameter
    # read fluid grid & construct fluid grid data


    #io_data = Io_Data("Test/NacaBF/Visc/naca.input")
    io_data = Io_Data("Test/Blasius/blasius.input")

    fluid_input = io_data.fluidmesh

    fluid = Fluid_Domain(fluid_input)
    #fluid.check_mesh(True)



    explicit_solver = Explicit_Solver(fluid,io_data)

    explicit_solver._solve()
    explicit_solver._draw_solution()

'''
#this is for embedded mesh simulation
from Embedded_Explicit_Solver_FIVER2 import *
def main():

    from Structure import Structure
    # get input parameter
    # read fluid grid & construct fluid grid data


    io_data = Io_Data("./Test/IBNaca/naca.input")
    fluid_input = io_data.fluidmesh
    structure_input = io_data.structuremesh

    fluid = Fluid_Domain(fluid_input)

    structure = Structure(structure_input)
    #fluid.check_mesh(True)



    explicit_solver = Embedded_Explicit_Solver(fluid,structure,io_data)

    explicit_solver._solve()
    explicit_solver._draw_solution()


    # if it is immersed boundary problem



    # read body grid & construct body grid data

    # construct intersector

    # set_initial_solution

main()
