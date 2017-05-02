__author__ = 'zhengyuh'

#read gmsh then convert and output .fgrid

import numpy as np
import argparse
import sys

class Fluid_Gmsh_Converter:
    def __init__(self, mshfile):

        self.meshname = mshfile
        mshfile += ".msh"

        ## verts, elmts, edges are numpy.array

        self.triangles = np.empty(shape=[0, 3],dtype=int)
        self.ntriangles = 0;

        self.quads = np.empty(shape=[0, 4],dtype=int)
        self.nquads = 0;


        self.edges = np.empty(shape=[0, 4],dtype=int)
        self.nedges = 0;

        ##define boundary as a dictionary of sets (a,b)
        self.boundary = []

        self.phys = {}
        self.phys_map = {}
        self.nphys = 0



        try:
            fid = open(mshfile, "r")
        except IOError:
            print("File '%s' not found." % mshfile)
            sys.exit()

        line = 'start'
        while line:
            line = fid.readline()

            if line.find('$MeshFormat') == 0:
                line = fid.readline()
                if line.split()[0][0] is not '2':
                    print("wrong gmsh version")
                    sys.exit()
                line = fid.readline()
                if line.find('$EndMeshFormat') != 0:
                    raise ValueError('expecting EndMeshFormat')

            if line.find('$PhysicalNames') == 0:
                line = fid.readline()
                self.nphys = int(line.split()[0])
                self.boundary = [set() for i in range(self.nphys)]
                for i in range(0, self.nphys):
                    line = fid.readline()
                    newkey = int(line.split()[1])
                    qstart = line.find('"')+1
                    qend = line.find('"', -1, 0)-1
                    self.phys_map[newkey] = i
                    self.phys[i]=line[qstart:qend]
                line = fid.readline()
                if line.find('$EndPhysicalNames') != 0:
                    raise ValueError('expecting EndPhysicalNames')

            if line.find('$Nodes') == 0:
                line = fid.readline()
                self.nverts = int(line.split()[0])
                self.verts = np.zeros((self.nverts, 2), dtype=float)
                for i in range(0, self.nverts):
                    line = fid.readline()
                    data = line.split()
                    self.verts[i, :] = list(map(float, data[1:3]))
                line = fid.readline()
                if line.find('$EndNodes') != 0:
                    raise ValueError('expecting EndNodes')




            if line.find('$Elements') == 0:
                line = fid.readline()
                nel = int(line.split()[0])
                for i in range(0, nel):
                    line = fid.readline()
                    data = line.split()
                    ntags = int(data[2])           # number of tags following
                    k = 3
                    if ntags > 0:                   # set physical id
                        physid = int(data[k])
                        k += ntags

                    nodes = list(map(int, data[k:]))
                    nodes = np.array(nodes,dtype=int)-1  # fixe gmsh 1-based index
                    nodes.sort()

                    if(int(data[1]) == 2): #triangle element
                        self.triangles = np.vstack((self.triangles, nodes))
                        self.ntriangles += 1;

                    elif(int(data[1]) == 3):#quad element
                        self.quads = np.vstack((self.quads, nodes))
                        self.nquads += 1;
                    else:#boundary line element
                        idx = int(data[0])-1  # fix gmsh 1-based indexing
                        if i != idx:
                            raise ValueError('problem with elements ids')

                        node_pair = (nodes[0], nodes[1])


                        self.boundary[self.phys_map[physid]].add(node_pair)


                line = fid.readline()
                if line.find('$EndElements') != 0:
                    raise ValueError('expecting EndElements')

        fid.close()
        self.edge_process()




    def edge_process(self):

        #construct a dictionary (n1,n2) to (e1,e2) or e
        edges_set = {}
        for e in range(self.ntriangles):
            [n1,n2,n3] = self.triangles[e,:]
            edge_pairs = [(n1,n2),(n1,n3),(n2,n3)]
            for edge_pair in edge_pairs:
                #n is the third node number

                if edge_pair in edges_set:
                    e1 = edges_set[edge_pair]
                    edges_set[edge_pair]  = (e1,e)
                else:
                    edges_set[edge_pair] = e
                    self.nedges += 1




        for e in edges_set:
            [n1,n2] = e
            for boundary_set in self.boundary:
                if e in boundary_set:
                    #check orientation
                    ele = edges_set[e]
                    n3 = sum(self.triangles[ele,:]) - n1 - n2
                    v1 = self.verts[n1,:]
                    v2 = self.verts[n2,:]
                    v3 = self.verts[n3,:]
                    if(np.cross(v2 - v1, v3-v1) < 0):
                        boundary_set.remove(e)
                        boundary_set.add((n2,n1))
                        del edges_set[e]
                        edges_set[(n2,n1)] = ele
                        break

        self.edges_set = edges_set






    def write(self, fname=None):
        if fname is None:
            fname = self.meshname + '.fgrid'
        if type(fname) is str:
            try:
                fid = open(fname, 'w')
            except IOError as e:
                (errno, strerror) = e.args
                print(".neu error (%s): %s" % (errno, strerror))
        else:
            raise ValueError('fname is assumed to be a string')

        #node number
        fid.write(' %10d \n' % self.nverts)
        #coordinates of node
        for i in range(self.nverts):
            fid.write(' %.16E %.16E \n' % (self.verts[i,0],self.verts[i,1]))
        #element number
        fid.write(' %10d \n' % self.ntriangles)
        for i in range(self.ntriangles):
            fid.write(' %10d %10d %10d \n' % (self.triangles[i,0],self.triangles[i,1],self.triangles[i,2]))
        #edge number
        fid.write(' %10d \n' % self.nedges)
        for key, ele in list(self.edges_set.items()):
            if(isinstance( ele, int )):
                fid.write(' %10d %10d %10d %10d\n' % (key[0], key[1], ele, -1))
            else:
                fid.write(' %10d %10d %10d %10d\n' % (key[0], key[1], ele[0], ele[1]))



        #boundary type number

        nbounds = 0
        for i in range(self.nphys):
            nbounds += len(self.boundary[i])
        fid.write(' %10d %10d\n' % (self.nphys ,nbounds))

        for i in range(self.nphys):

            fid.write("%10d %10s\n" % ( len(self.boundary[i]),self.phys[i]))
            for n1,n2 in self.boundary[i]:
                e = self.edges_set[(n1,n2)]
                fid.write("%10d %10d %10d\n" %(n1, n2, e))





class Surface_Gmsh_Converter:
    def __init__(self, mshfile):



        self.meshname = mshfile
        mshfile += ".msh"

        ## verts, elmts, edges are numpy.array

        ##define boundary as a dictionary of sets (a,b)
        self.boundary = []

        self.phys =[]
        self.nphys = 0

        try:
            fid = open(mshfile, "r")
        except IOError:
            print("File '%s' not found." % mshfile)
            sys.exit()

        line = 'start'
        while line:
            line = fid.readline()

            if line.find('$MeshFormat') == 0:
                line = fid.readline()
                if line.split()[0][0] is not '2':
                    print("wrong gmsh version")
                    sys.exit()
                line = fid.readline()
                if line.find('$EndMeshFormat') != 0:
                    raise ValueError('expecting EndMeshFormat')

            if line.find('$PhysicalNames') == 0:
                line = fid.readline()
                nphys = int(line.split()[0])
                if nphys != 1:
                    raise ValueError('physical element type error')

                fid.readline()

                line = fid.readline()
                if line.find('$EndPhysicalNames') != 0:
                    raise ValueError('expecting EndPhysicalNames')

            if line.find('$Nodes') == 0:
                line = fid.readline()
                self.nverts = int(line.split()[0])
                self.verts = np.zeros((self.nverts, 2), dtype=float)
                for i in range(0, self.nverts):
                    line = fid.readline()
                    data = line.split()
                    self.verts[i, :] = list(map(float, data[1:3]))
                line = fid.readline()
                if line.find('$EndNodes') != 0:
                    raise ValueError('expecting EndNodes')

            if line.find('$Elements') == 0:
                line = fid.readline()
                self.nbounds = int(line.split()[0])
                self.bounds = np.empty(shape=[self.nbounds,2],dtype=int)
                for i in range(self.nbounds):
                    line = fid.readline()
                    data = line.split()
                    ntags = int(data[2])           # number of tags following
                    k = 3
                    if ntags > 0:                   # set physical id
                        physid = int(data[k])
                        k += ntags

                    nodes = list(map(int, data[k:]))
                    nodes = np.array(nodes,dtype=int)-1  # fixe gmsh 1-based index


                    if(int(data[1]) != 1): #triangle element
                        raise ValueError('not boundary element')
                    else:#boundary line element
                        idx = int(data[0])-1  # fix gmsh 1-based indexing
                        if i != idx:
                            raise ValueError('problem with elements ids')


                        self.bounds[i,:] = nodes


                line = fid.readline()
                if line.find('$EndElements') != 0:
                    raise ValueError('expecting EndElements')

        fid.close()








    def write(self, fname=None):
        if fname is None:
            fname = self.meshname + '.sgrid'
        if type(fname) is str:
            try:
                fid = open(fname, 'w')
            except IOError as e:
                (errno, strerror) = e.args
                print(".neu error (%s): %s" % (errno, strerror))
        else:
            raise ValueError('fname is assumed to be a string')

        #node number
        fid.write(' %10d \n' % self.nverts)
        #coordinates of node
        for i in range(self.nverts):
            fid.write(' %.16E %.16E \n' % (self.verts[i,0],self.verts[i,1]))
        #boundary type number

        fid.write(' %10d \n' % (self.nbounds))

        for n1,n2 in self.bounds:
                fid.write("%10d %10d \n" %(n1, n2))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="convert fluid mesh or structure mesh")
    parser.add_argument("-s", "--structure", type=str, help="convert structure mesh, name without suffix")
    parser.add_argument("-f", "--fluid", type=str, help="convert fluid mesh, name without suffix")
    args = parser.parse_args()
    if args.fluid:
        mesh = Fluid_Gmsh_Converter(args.fluid)
        mesh.write();

    if args.structure:
        mesh = Surface_Gmsh_Converter(args.structure)
        mesh.write();
  
    







