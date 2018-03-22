#!/usr/bin/python
# coding: utf-8
"""
Decomposing an assembly obtained from a STEP file

"""

from __future__ import print_function
import wx
import os
import pdb
import logging
from os import path as _path
from osvcad.nodes import AssemblyGeometryNode
import ccad.model as cm
import ccad.display as cd
from aocxchange.step import StepImporter
from aocutils.display.wx_viewer import Wx3dViewer
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import ccad.model as ccm
import pointcloud as pc
logger = logging.getLogger('__name__')

class Assembly(nx.DiGraph):
    r""" Assembly Class

    This class has to be connected to the osvcad AssemblyGeometry representation

    An Assembly is a Graph.
    Each node of an Assembly represents a solid or an Assembly
    Each node of an Assembly is described in an external file
    A node has the following attributes :
    'name' : name of the origin file (step file or python file)
        'V' : a unitary matrix
        'dim' : a dimension integer
        'ptc' : a translation vector

    Methods
    -------

    from_step
    tag_nodes
    write_components
    __repr__

    """
    def __init__(self, shape, origin=None, direct=False):
        """
        Parameters
        ----------

        shape :
        origin : str
            The file or script the assembly was created from
        direct : bool, optional(default is False)
            If True, directly uses the point cloud of the Shell
            If False, iterate the faces, wires and then vertices

        """

        super(Assembly,self).__init__()
        self.shape = shape
        #self.G = nx.DiGraph()
        self.pos = dict()
        self.origin = origin

        shells = self.shape.subshapes("Shell")
        logger.info("%i shells in assembly" % len(shells))
        for k, shell in enumerate(shells):
            logger.info("Dealing with shell nb %i" % k)
            #pcloud = np.array([[]])
            #pcloud.shape = (3, 0)
            pcloud = pc.PointCloud()

            if direct:
                vertices = shell.subshapes("Vertex")
                logger.info("%i vertices found for direct method")
                for vertex in vertices:
                    point = np.array(vertex.center())[:, None]
                    #pcloud = np.append(pcloud, point, axis=1)
                    pcloud = pcloud + point
            else:
                # get faces from shell
                faces = shell.subshapes("Face")

                for face in faces:
                    face_type = face.type()
                    wires = face.subshapes("Wire")

                    for wire in wires:
                        vertices = wire.subshapes("Vertex")

                        for vertex in vertices:
                            point = np.array(vertex.center())[:, None]
                            #pcloud = np.append(pcloud, point, axis=1)
                            pcloud = pcloud + point

                    if face_type == "plane":
                        pass
                    if face_type == "cylinder":
                        pass

            # add shape to graph if shell not degenerated
            Npoints = pcloud.p.shape[0]

            if ((shell.area()>0) and Npoints >=3):
                self.add_node(k, pcloud=pcloud, shape=shell)
                self.pos[k] = shell.center()

    def view_graph(self,**kwargs):
        """ view an assembly graph

        Parameters
        ----------

        fontsize
        v
        bshow
        blabels
        alpha
        figsize

        """

        fontsize=kwargs.pop('fontsize', 18)
        v = kwargs.pop('v', 20)
        bsave = kwargs.pop('bsave', False)
        bshow= kwargs.pop('bshow', True)
        blabels = kwargs.pop('blabels', False)
        alpha = kwargs.pop('alpha', 0.5)
        figsize= kwargs.pop('figsize', (6,6))

        dxy = { k : (self.pos[k][0],self.pos[k][1]) for k in self.node.keys() }
        dxyl = { k : (self.pos[k][0]+(v*np.random.rand()-v/2.),self.pos[k][1]+(v*np.random.rand()-v/2.)) for k in self.node.keys() }
        dxz = { k : (self.pos[k][0],self.pos[k][2]) for k in self.node.keys() }
        dxzl = { k : (self.pos[k][0]+(v*np.random.rand()-v/2),self.pos[k][2]+(v*np.random.rand()-v/2.)) for k in self.node.keys() }
        dyz = { k : (self.pos[k][2],self.pos[k][1]) for k in self.node.keys() }
        dyzl = { k : (self.pos[k][2]+(v*np.random.rand()-v/2),self.pos[k][1]+(v*np.random.rand()-v/2.)) for k in self.node.keys() }
        #node_size = [ self.node[k]['dim'] for k in self.node.keys() ]
        node_size = 10

        #dlab = {k : str(int(np.ceil(self.node[k]['dim']))) for k in self.node.keys() if self.edge[k].keys()==[] }
        dlab = {k : self.node[k]['pcloud'].sig for k in self.node.keys() }

        plt.figure(figsize=figsize)
        plt.suptitle(self.origin,fontsize=fontsize+2)
        plt.subplot(2,2,1)
        nx.draw_networkx_nodes(self,dxy,node_size=node_size,alpha=alpha)
        nx.draw_networkx_edges(self,dxy)
        if blabels:
            nself.draw_networkx_labels(self,dxyl,labels=dlab,font_size=fontsize)
        plt.xlabel('X axis (mm)',fontsize=fontsize)
        plt.ylabel('Y axis (mm)',fontsize=fontsize)
        plt.title("XY plane",fontsize=fontsize)
        plt.subplot(2,2,2)
        nx.draw_networkx_nodes(self,dyz,node_size=node_size,alpha=alpha)
        nx.draw_networkx_edges(self,dyz)
        if blabels:
            nx.draw_networkx_labels(self,dyzl,labels=dlab,font_size=fontsize)
        plt.xlabel('Z axis (mm)',fontsize=fontsize)
        plt.ylabel('Y axis (mm)',fontsize=fontsize)
        plt.title("ZY plane",fontsize=fontsize)
        plt.subplot(2,2,3)
        nx.draw_networkx_nodes(self,dxz,node_size=node_size,alpha=alpha)
        nx.draw_networkx_edges(self,dxz)
        if blabels:
            nx.draw_networkx_labels(self,dxzl,labels=dlab,font_size=fontsize)
        plt.title("XZ plane",fontsize=fontsize)
        plt.xlabel('X axis (mm)',fontsize=fontsize)
        plt.ylabel('Z axis (mm)',fontsize=fontsize)
        plt.subplot(2,2,4)
        if blabels:
            nx.draw(self,labels=dlab,alpha=alpha,font_size=fontsize,node_size=node_size)
        else:
            nx.draw(self,alpha=alpha,font_size=fontsize,node_size=node_size)
        if bsave:
            plt.savefig(self.origin+'png')
        if bshow:
            plt.show()

    def __repr__(self):
        #st = self.shape.__repr__()+'\n'
        for k in self.node:
            st += self.node[k]['pcloud'].name  +'\n'
        return st

    @classmethod
    def from_step(cls, filename, direct=False):
        r"""Create an Assembly instance from a STEP file

        Parameters
        ----------

        filename : str
            path to the STEP file
        direct : bool, optional(default is False)
            If True, directly use the point cloud of the Shell
            If False, iterate the faces, wires and vertices

        Returns
        -------
        Assembly : the new Assembly object created from a STEP file

        """
        solid = cm.from_step(filename)
        return cls(solid, origin=filename, direct=direct)

    def tag_nodes(self):
        r"""Add computed data to each node of the assembly

        self.node
            dim
            name
            ptc
            pcloud
            shape
            V


        """
        # self.lsig : list of signatures
        #
        # iterate over nodes
        #    iterate over lower nodes
        #       check if point cloud are equal
        #       check if point cloud are close
        # dist is the distance fingerprint
        self.lsig = []
        for k in self.node:
             pcloudk = self.node[k]['pcloud']
             mink = np.max(pcloudk.p,axis=0)
             maxk = np.max(pcloudk.p,axis=0)
             dk = pcloudk.dist
             for j in range(k):
                pj = self.node[k]['pcloud']
                dj = self.node[j]['dist']
                if len(dk)==len(dj):
                    Edn = np.sum(dk)
                    Edj = np.sum(dj)
                    rho1 = np.abs(Edn-Edj)/(Edn+Edj)
                    DEjk = np.sum(np.abs(dk-dj))
                    rho2 = DEjk/(Edn+Edj)
                    #
                    # Relation 1 : equal
                    #
                    if np.allclose(DEjk,0):
                        # The two point clouds are equal w.r.t sorted points to origin distances
                        if self.edge[j].keys()==[]:
                            self.add_edge(k,j,equal=True,close=True)
                    #
                    # Relation 2 : almost equal
                    #
                    elif (rho1<0.01) and (rho2<0.05):
                        if self.edge[j].keys()==[]:
                        # The two point clouds are closed w.r.t sorted point to origin distances
                            self.add_edge(k,j,equal=False,close=True)


        #
        # once all edges are informed
        #
        self.lsig = []
        for k in self.node:
            pcloudk = self.node[k]['pcloud']

            lsamek = [ x for x in self.edge[k].keys() if self.edge[k][x]['equal']]

            if lsamek==[]:
                self.lsig.append(sig)
                self.node[k]['name'] = pcloudk.sig
                # self.node[k]['V'] = V
                # self.node[k]['dim'] = dim
            else:
                refnode = [x for x in lsamek if self.edge[x].keys()==[]][0]
                self.node[k]['name'] = self.node[refnode]['name']
                pcsame = self.node[refnode]['pc']

                # self.node[k]['V']= self.node[refnode]['V']
                # self.node[k]['dim']= self.node[refnode]['dim']
                #
                # detection of eventual symmetry
                #
                # The symmetry is informed in the node
                #

                dp = np.sum(np.abs(pcsame-pcloudk.pc),axis=1)
                nomirror = np.isclose(dp,0)
                if nomirror[0]==False:
                    self.add_node(k,mx=True)
                if nomirror[1]==False:
                    self.add_node(k,my=True)
                if nomirror[2]==False:
                    self.add_node(k,mz=True)

            self.node[k]['V'] = V
            self.node[k]['pc'] = pc
            self.node[k]['dim'] = int(np.ceil(dim))


        self.lsig = list(set(self.lsig))
        self.Nn = len(self.node)

    def clean(self):
        """
        Clean temporary data before serializing the graph
        """
        for (n,d) in self.nodes(data=True):
            del d['pcloud']
            del d['shape']
            del d['dist']
            del d['pc']

        # set a boolean for not cleaning twice
        self.bclean = True

    def serialize(self):
        """ serialize matrix in assembly

        Notes
        -----

        iterates on nodes
        get unitary matrix V and ravels it

        """
        for (n,d) in self.nodes(data=True):
            V = d['V']
            ptc = d['ptc']
            lV = str(list((d['V'].ravel())))
            lptc = str(list((d['ptc'])))
            ptcr = np.array(eval(lptc))
            Vr = np.array(eval(lV)).reshape(3,3)
            assert(np.isclose(V-Vr,0).all())
            assert(np.isclose(ptc-ptcr,0).all())
            d['V']=lV
            d['ptc']=lptc
        self.serialized=True

    def unserialize(self):
        """ unserialize matrix in assembly

        Notes
        -----

        In the gml or json file the 3x3 matrix is stored as a line
        this function recover the matrix form

        """

        for (n,d) in self.nodes(data=True):
            lV = d['V']
            lptc = d['ptc']
            ptcr = np.array(eval(lptc))
            Vr = np.array(eval(lV)).reshape(3,3)
            d['V']=Vr
            d['ptc']=ptcr
        self.serialized=False

    def save_json(self):
        if not self.bclean:
            self.clean
        self.serialize()
        data = json_graph.node_link_data(self)
        filename = self.origin.replace('.stp','.json')
        fd = open(filename,'w')
        with fd:
            json.dump(data,fd)
        self.unserialize()

    def load_json(self,filename):
        """ load Assembly from json file
        """
        fd = open(filename,'r')
        data = json.load(fd)
        fd.close()
        G = json_graph.node_link_graph(data,directed=True)
        self.nodes = G.nodes
        self.edges = G.edges
        self.node = G.node
        self.edge = G.edge
        self.origin = filename
        self.unserialize()
        for inode in self:
            self.pos[inode] = self.node[inode]['ptc']

    def save_gml(self):
        if not self.bclean:
            self.clean()
        self.serialize()
        filename = self.origin.replace('.stp','.gml')
        nx.write_gml(self,filename)
        self.unserialize()

    def write_components(self):
        r"""Write components of the assembly

        Notes
        -----

        Write components to their own step files in a
        subdirectory of the folder containing the original file

        """

        if os.path.isfile(self.origin):
            directory = os.path.dirname(self.origin)
            basename = os.path.basename(self.origin)
            subdirectory = os.path.join(directory,
                                        os.path.splitext(basename)[0])
            if not os.path.isdir(subdirectory):
                os.mkdir(subdirectory)
        else:
            msg = "The components of the assembly should already exist"
            raise ValueError(msg)

        for k in self.node:
            # calculate point cloud signature
            self.node[k]['pcloud'].signature()

            name = self.node[k]['pcloud'].name
            sig = self.node[k]['pcloud'].sig
            vec = self.node[k]['pcloud'].vec
            ang = self.node[k]['pcloud'].ang
            ptm = self.node[k]['pcloud'].ptm

            shp = self.node[k]['shape']
            filename = name + ".stp"
            if not os.path.isfile(filename):
                shp.translate(-ptm)
                if abs(ang)>0:
                    shp.rotate(np.array([0, 0, 0]), vec, ang)
                filename = os.path.join(subdirectory, filename)
                shp.to_step(filename)


def reverse(step_filename, view=False):
    r"""Reverse STEP file using ccad

    Parameters
    ----------

    step_filename : str
        Path to the STEP file
    view : bool, optional (default is False)
        Launch the ccad viewer?

    """

    # read a step file
    assembly = Assembly.from_step(step_filename, direct=False)
    # decompose assembl in elementary parts
    assembly.write_components()

    assembly.tag_nodes()

    if view:
        ccad_viewer = cd.view()
        for shell in assembly.shape.subshapes("Shell"):
            ccad_viewer.display(shell)
        cd.start()

    return(assembly)


def view(step_filename):
    r"""View the STEP file contents in the aocutils wx viewer.

    The aocutils wx viewer is good to visualize topology.

    Parameters
    ----------

    step_filename : str
        path to the STEP file

    """

    importer = StepImporter(filename=step_filename)

    class MyFrame(wx.Frame):
        r"""Frame for testing"""
        def __init__(self):
            wx.Frame.__init__(self, None, -1)
            self.p = Wx3dViewer(self)
            for shape in importer.shapes:
                self.p.display_shape(shape)
            self.Show()

    app = wx.App()
    frame = MyFrame()
    app.SetTopWindow(frame)
    app.MainLoop()

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s :: %(levelname)6s :: '
                               '%(module)20s :: %(lineno)3d :: %(message)s')
    #filename = "../step/0_tabby2.stp"  # OCC compound
    #filename = "../step/ASM0001_ASM_1_ASM.stp"  # OCC compound
    # filename = "../step/MOTORIDUTTORE_ASM.stp" # OCC compound
    filename = "../step/aube_pleine.stp"  # OCC Solid

    a1 = reverse(filename)
    #cd.view(a1)
