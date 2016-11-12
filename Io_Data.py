__author__ = 'zhengyuh'

class Io_Data():
    def __init__(self,input_file):



        with open(input_file) as fid:
            for line in fid:
                 lines = line.split()
                 if(len(lines) < 2):
                     break

                 if(lines[0] == "fluidmesh"):
                     self.fluidmesh = lines[1]

                 elif(lines[0] == "framework"):
                     self.frame = lines[1]

                 elif(lines[0] == "problem_type"):
                     self.problem_type = lines[1]

                 elif(lines[0] == "equation_type"):
                     self.equation_type = lines[1]

                 elif(lines[0] == "time_integration"):
                     self.time_integration = lines[1]

                 elif(lines[0] == "space_integration"):
                     self.space_integration = lines[1]
                 elif(lines[0] == "limiter"):
                     self.limiter = lines[1]


                 elif(lines[0] == "gamma"):
                     self.gamma = float(lines[1])
                 elif(lines[0] == "Prandtl"):
                     self.Prandtl = float(lines[1])

                 elif(lines[0] == "Reynolds"):
                     self.Reynolds= float(lines[1])

                 elif(lines[0] == "Mach"):
                     self.Mach = float(lines[1])
                 elif(lines[0] == "AoA"):
                     self.AoA = float(lines[1])
                 elif(lines[0] == "tolerance"):
                     self.tolerance = float(lines[1])
                 elif(lines[0] == "max_ite"):
                     self.max_ite = int(lines[1])

                 else:
                     print("'%s' not recognized error in inputfile" %lines[0])




'''


        #framework
        embdedded
        body_fitted

        #problem_type
        unsteady
        steady

        #equation_type
        Euler
        Navier_Stokes


        #time_integration
        implicit
        explicit

        #space_integration
        RK2

        #limitor
        0
        1
        Van_Albada

        #fluid_condition
        gamma
        Prandtl
        R

        #boundary_condition
        free_stream
        density
        pressure
        mach
        angle

        isothermal_wall

        slip_wall

        adiabatic_wall
'''
