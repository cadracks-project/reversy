#!/usr/bin/python
# coding: utf-8

"""Decomposing an assembly obtained from a STEP file
"""

from __future__ import print_function
# import wx
import os
# import time
# import pdb
import logging
import json
import networkx as nx
from networkx.readwrite import json_graph
# from os import path as _path
# from osvcad.nodes import AssemblyGeometryNode

import ccad.model as cm
import ccad.display as cd
from OCC.Display.WebGl import threejs_renderer, jupyter_renderer
# from aocxchange.step import StepImporter
# from aocutils.display.wx_viewer import Wx3dViewer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    from interval import interval
except ImportError:
    from interval import Interval as interval

from . import pointcloud as pc

logger = logging.getLogger(__name__)


class Assembly(nx.DiGraph):
    r""" Assembly Class

    This class has to be connected to the osvcad Assembly Geometry representation

    An Assembly is a Graph.

    Each node of an Assembly represents a solid or an Assembly

    Each node of an Assembly is described in an external file

    An assembly has the following members

    nnodes : int
        number of nodes

    A node has the following attributes :

    'name' : name of the origin file (step file or python file)
    'pc' : a translation vector
    'V' : a unitary matrix
    'volume' : volume of the solid associated to the node
    'assembly' : boolean
        Tuue if the node is an assembly

    Methods
    -------

    from_step
    tag_nodes
    write_components
    __repr__

    """
    def __init__(self):
        super(Assembly, self).__init__()
        self.pos = dict()
        self.serialized = False

    def from_step(self, filename):
        """ creates an non hierarchical assembly with a solid per node

        Parameters
        ----------
        filename : str
            step file name

        Notes
        -----

        Creates a node per valid solid (check TRUE)
        This assembly is not clean it contains a lot of information in each
        node.
        pcloud : pc.Pointcloud (centered and ordered point cloud)
        shape : cm.Solid

        """
        self.solid = cm.from_step(filename)
        self.isclean = False
        #
        # if it contains, An Assembly:
        # is clean : filename and transformation
        # is not clean : pointcloud and shape
        #
        # self.G = nx.DiGraph()
        self.origin = filename
        shells = self.solid.subshapes("Shell")
        # logger.info("%i shells in assembly" % len(shells))
        self.nnodes = 0
        for k, shell in enumerate(shells):
            solid = cm.Solid([shell])
            # check the shell corresponds to a cosed solid
            if solid.check():
                # logger.info("Dealing with shell nb %i" % k)
                # pcloud = np.array([[]])
                # pcloud.shape = (3, 0)
                # pcloud = pc.PointCloud()
                # pcloud = pcloud.from_solid(solid)
                # vertices = shell.subshapes("Vertex")
                # logger.info("%i vertices found for direct method")
                # for vertex in vertices:
                #     point = np.array(vertex.center())[:, None]
                    # pcloud = np.append(pcloud, point, axis=1)
                #     pcloud = pcloud + point
                # add shape to graph if shell not degenerated
                # Npoints = pcloud.p.shape[0]

                # if ((shell.area()>0) and Npoints >=3):
                if shell.area() > 0:
                    # pcloud.centering()
                    # pcloud.ordering()
                    # the stored point cloud is centered and ordered
                    # self.add_node(self.nnodes,
                    #               pcloud=pcloud,
                    #               shape=solid,
                    #               volume=solid.volume(),
                    #               assembly=False)
                    self.add_node(self.nnodes,
                                  shape=solid,
                                  volume=solid.volume(),
                                  assembly=False)
                    self.pos[self.nnodes] = solid.center()
                    self.nnodes += 1

    def __repr__(self):
        # st = self.shape.__repr__()+'\n'
        st = str(self.nnodes) + ' nodes' + '\n'
        for k in self.node:
            st += self.node[k]['name'] + '\n'
        return st

    def remove_nodes(self, lnodes):
        assert(x in self.node for x in lnodes)
        self.remove_nodes_from(lnodes)
        [self.pos.pop(x) for x in lnodes ]
        self.nnodes = len(self.node)

    def show_graph(self, **kwargs):
        """ show an assembly graph

        Parameters
        ----------
        fontsize
        v
        bshow
        blabels
        alpha
        figsize

        """
        fontsize = kwargs.pop('fontsize', 18)
        v = kwargs.pop('v', 20)
        bsave = kwargs.pop('bsave', False)
        bshow = kwargs.pop('bshow', True)
        blabels = kwargs.pop('blabels', False)
        alpha = kwargs.pop('alpha', 0.5)
        figsize = kwargs.pop('figsize', (6, 6))

        dxy = {k: (self.pos[k][0],
                   self.pos[k][1]) for k in self.node.keys()}
        dxyl = {k: (self.pos[k][0]+(v*np.random.rand()-v/2.),
                    self.pos[k][1]+(v*np.random.rand()-v/2.)) for k in self.node.keys()}
        dxz = {k: (self.pos[k][0],
                   self.pos[k][2]) for k in self.node.keys()}
        dxzl = {k: (self.pos[k][0]+(v*np.random.rand()-v/2),
                    self.pos[k][2]+(v*np.random.rand()-v/2.)) for k in self.node.keys()}
        dyz = {k: (self.pos[k][2],
                   self.pos[k][1]) for k in self.node.keys()}
        dyzl = {k: (self.pos[k][2]+(v*np.random.rand()-v/2),
                    self.pos[k][1]+(v*np.random.rand()-v/2.)) for k in self.node.keys()}
        node_size = [self.node[k]['volume']/1000. for k in self.node.keys()]

        dlab = {k: self.node[k]['sig'] for k in self.node.keys()}

        plt.figure(figsize=figsize)
        plt.suptitle(self.origin, fontsize=fontsize+2)
        plt.subplot(2, 2, 1)

        lequal = [x for x in self.edges() if self.edge[x[0]][x[1]]['equal']]
        lsim = [x for x in self.edges() if self.edge[x[0]][x[1]]['sim']]
        lintersect = [x for x in self.edges() if self.edge[x[0]][x[1]]['intersect']]
        lclose = [x for x in self.edges() if self.edge[x[0]][x[1]]['close']]
        # lequal = [ x for x in self.edges()
        #                                   if 'equal' in self.edge[x[0]][x[1]]]
        # print(lequal)
        # print(lsim)
        # print(lintersect)
        # print(lclose)
        nx.draw_networkx_nodes(self, dxy, node_size=node_size, alpha=alpha)
        nx.draw_networkx_edges(self, dxy, edgelist=lequal, edge_color='b')
        nx.draw_networkx_edges(self, dxy, edgelist=lsim, edge_color='c')
        nx.draw_networkx_edges(self, dxy, edgelist=lintersect, edge_color='r')
        nx.draw_networkx_edges(self, dxy, edgelist=lclose, edge_color='m')

        # edgelist_close = [ (x,y) for (x,y) in self.edges()
        #                                           if self.edge[x][y]['close']]
        # edgelist_intersect = [ (x,y) for (x,y) in self.edges()
        #                                       if self.edge[x][y]['intersect']]

        if blabels:
            nx.draw_networkx_labels(self, dxyl, labels=dlab, font_size=fontsize)

        plt.xlabel('X axis (mm)', fontsize=fontsize)
        plt.ylabel('Y axis (mm)', fontsize=fontsize)
        plt.title("XY plane", fontsize=fontsize)
        plt.subplot(2, 2, 2)

        nx.draw_networkx_nodes(self, dyz, node_size=node_size, alpha=alpha)
        nx.draw_networkx_edges(self, dyz, edgelist=lequal, edge_color='b')
        nx.draw_networkx_edges(self, dyz, edgelist=lsim, edge_color='c')
        nx.draw_networkx_edges(self, dyz, edgelist=lintersect, edge_color='r')
        nx.draw_networkx_edges(self, dyz, edgelist=lclose, edge_color='m')
        if blabels:
            nx.draw_networkx_labels(self, dyzl, labels=dlab, font_size=fontsize)
        plt.xlabel('Z axis (mm)', fontsize=fontsize)
        plt.ylabel('Y axis (mm)', fontsize=fontsize)
        plt.title("ZY plane", fontsize=fontsize)

        plt.subplot(2, 2, 3)
        nx.draw_networkx_nodes(self, dxz, node_size=node_size, alpha=alpha)

        nx.draw_networkx_edges(self, dxz, edgelist=lequal, edge_color='b')
        nx.draw_networkx_edges(self, dxz, edgelist=lsim, edge_color='c')
        nx.draw_networkx_edges(self, dxz, edgelist=lintersect, edge_color='r')
        nx.draw_networkx_edges(self, dxz, edgelist=lclose, edge_color='m')

        if blabels:
            nx.draw_networkx_labels(self, dxzl, labels=dlab, font_size=fontsize)
        plt.title("XZ plane", fontsize=fontsize)
        plt.xlabel('X axis (mm)', fontsize=fontsize)
        plt.ylabel('Z axis (mm)', fontsize=fontsize)
        plt.subplot(2, 2, 4)
        plt.tight_layout()
        if blabels:
            nx.draw(self,
                    labels=dlab,
                    alpha=alpha,
                    font_size=fontsize,
                    node_size=node_size)
        else:
            nx.draw(self,
                    alpha=alpha,
                    font_size=fontsize,
                    node_size=node_size)
        if bsave:
            plt.savefig(self.origin+'png')
        if bshow:
            plt.show()

    def equalsim_nodes_edges(self):
        r""" connect equal and sim nodes

        self.node
            dim
            name
            pc
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
        if not self.isclean:
            self.df_edges = pd.DataFrame(columns=('tail', 'head', 'equal', 'sim', 'intersect', 'close'))
            self.lsig = []
            for k in self.node:
                solidk = self.node[k]['shape']
                pcloudk = pc.PointCloud()
                pcloudk = pcloudk.from_solid(solidk)
                pcloudk.sorting()
                pcloudk.ordering()
                # mink = np.min(pcloudk.p, axis=0)
                # maxk = np.max(pcloudk.p, axis=0)
                dk = pcloudk.dist
                for j in range(k):
                    solidj = self.node[j]['shape']
                    pcloudj = pc.PointCloud()
                    pcloudj = pcloudj.from_solid(solidj)
                    pcloudj.sorting()
                    pcloudj.ordering()

                    # print(k,j,dint[~bint])
                    dj = pcloudj.dist
                    # same number of points
                    if len(dk) == len(dj):
                        Edn = np.sum(dk)
                        Edj = np.sum(dj)
                        rho1 = np.abs(Edn-Edj)/(Edn+Edj)
                        DEjk = np.sum(np.abs(dk-dj))
                        rho2 = DEjk/(Edn+Edj)
                        #
                        # Relation 1 : equal
                        #
                        if np.allclose(DEjk, 0):
                            # The two point clouds are equal w.r.t sorted
                            # points to origin distances
                            if self.edge[j].keys() == []:
                                self.add_edge(k, j,
                                              equal=True,
                                              sim=True,
                                              intersect=False,
                                              close=False,
                                              djk=DEjk)
                        #
                        # Relation 2 : almost equal
                        #
                        elif (rho1 < 0.01) and (rho2 < 0.05):
                            if self.edge[j].keys() == []:
                                # The two point clouds are closed w.r.t sorted
                                # point to origin distances
                                self.add_edge(k, j,
                                              equal=False,
                                              sim=True,
                                              intersect=False,
                                              close=False)

            self.lsig = []
            for k in self.node:
                solidk = self.node[k]['shape']
                pcloudk = pc.PointCloud()
                pcloudk = pcloudk.from_solid(solidk)
                pcloudk.sorting()
                pcloudk.ordering()
                pcloudk.signature()

                lsamek = [x for x in self.edge[k].keys() if self.edge[k][x]['equal']]

                if lsamek == []:
                    self.lsig.append(pcloudk.sig)
                    # self.node[k]['name'] = pcloudk.name
                    # self.node[k]['V'] = pcloudk.V
                else:
                    refnode = [x for x in lsamek if self.edge[x].keys()==[]][0]
                    self.node[k]['name'] = self.node[refnode]['name']
                    pcsame = self.node[refnode]['pc']
                    #
                    # self.node[k]['V']= self.node[refnode]['V']
                    #
                    # detection of eventual symmetry
                    #
                    # The symmetry is informed in the node
                    #
                    vec = np.abs(pcsame-pcloudk.pc)[None,:]
                    dp = np.sum(vec, axis=0)
                    nomirror = np.isclose(dp, 0)
                    if nomirror[0] is False:
                        self.add_node(k, mx=True)
                    if nomirror[1] is False:
                        self.add_node(k, my=True)
                    if nomirror[2] is False:
                        self.add_node(k, mz=True)

                # self.node[k]['V'] = pcloudk.V
                # self.node[k]['pc'] = pcloudk.pc

        # unique the list
        self.lsig = list(set(self.lsig))
        self.Nn = len(self.node)

    def delete_edges(self,kind='equal'):
        """ delete edges of a specific kind

        Parameters
        ----------
        kind : string

        """

        for ed in self.edges():
            # if self.edge[ed[0]][ed[1]].has_key(kind):
            if kind in self.edge[ed[0]][ed[1]]:
                self.remove_edge(ed[0], ed[1])

    def intersect_nodes_edges(self, tol=2):
        r"""Intersect nodes edges

        Parameters
        ----------
        tol

        """
        for k in self.node:
            solidk = self.get_solid_from_nodes([k])
            for j in range(k):
                solidj = self.get_solid_from_nodes([j])
                bint, dint = intersect(solidk, solidj)
                dist = dint[~bint]
                if len(dist) == 0:
                    # print(k, j, dist)
                    self.add_edge(k, j,
                                  intersect=True,
                                  close=True,
                                  equal=False,
                                  sim=False)
                elif len(dist) == 1:
                    # print(dist[0])
                    if dist[0] < tol:
                        self.add_edge(k, j,
                                      close=True,
                                      intersect=False,
                                      equal=False,
                                      sim=False)

    def clean(self):
        """
        Clean temporary data before serializing the graph
        """
        for (n, d) in self.nodes(data=True):
            del d['shape']
            del d['pcloud']

        # set a boolean for not cleaning twice
        self.bclean = True

    def serialize(self):
        """ serialize matrix in assembly

        Notes
        -----

        iterates on nodes
        get unitary matrix V and ravels it

        """
        for (n, d) in self.nodes(data=True):
            V = d['V']
            ptc = d['pc']
            lV = str(list((d['V'].ravel())))
            lpc = str(list((d['pc'])))

            pcr = np.array(eval(lpc))
            Vr = np.array(eval(lV)).reshape(3, 3)

            assert(np.isclose(V - Vr, 0).all())
            assert(np.isclose(ptc - pcr, 0).all())

            d['V'] = lV
            d['pc'] = lpc

        self.serialized = True

    def unserialize(self):
        """ unserialize matrix in assembly

        Notes
        -----

        In the gml or json file the 3x3 matrix is stored as a line
        this function recover the matrix form

        """

        for (n, d) in self.nodes(data=True):
            lV = d['V']
            lptc = d['pc']
            ptcr = np.array(eval(lptc))
            Vr = np.array(eval(lV)).reshape(3,3)
            d['V'] = Vr
            d['pc'] = ptcr
        self.serialized = False

    def save_json(self, filename=''):
        """ save Assembly in json format

        Parameters
        ----------
        filename : string

        Notes
        -----
        If filename=='' filename is constructed from the origin file, i.e
        the step file which is at teh beginning of the analysis

        SEE ALSO
        --------

        networkx.readwrite.json_graph
        json.dump

        """
        if not self.isclean:
            self.clean()
            self.isclean = True
        self.serialize()

        data = json_graph.node_link_data(self)

        # filename construction
        rep = os.path.dirname(self.origin)
        basename = os.path.basename(self.origin)
        rep = os.path.join(rep, os.path.splitext(basename)[0])
        if filename == '':
            filename = os.path.splitext(basename)[0]+'.json'
        filename = os.path.join(rep, filename)

        fd = open(filename, 'w')
        with fd:
            json.dump(data, fd)
        self.unserialize()

    def from_json(self, filename):
        """ load Assembly from json file

        Parameters
        ----------
        filename : string

        """
        # read Graph data
        fd = open(filename)
        data = json.load(fd)
        fd.close()

        G = json_graph.node_link_graph(data, directed=True)

        self.isclean = True
        self.nodes = G.nodes
        self.edges = G.edges
        self.node = G.node
        self.edge = G.edge
        self.origin = filename

        # transform string node data in numpy.array
        self.unserialize()

        # update nodes pos in graph
        for inode in self:
            self.pos[inode] = self.node[inode]['pc']

    def merge_nodes(self, lnodes):
        """ merge a list of nodes into a sub Assembly

        Parameters
        ----------
        lnodes : list of nodes

        Returns
        -------
        A : Assembly
            a sub assembly

        """
        G = self.subgraph(lnodes)
        pos = {x: self.pos[x] for x in lnodes}
        # transcode node number
        lnodes_t = [self.dnodes[x] for x in lnodes]
        df_nodes = self.df_nodes.loc[lnodes_t]

        A = Assembly()
        A.add_nodes_from(G.nodes())
        A.add_edges_from(G.edges())
        A.node = G.node
        A.edge = G.edge
        A.pos = pos
        A.origin = self.origin
        A.isclean = self.isclean
        A.nnodes = len(lnodes)

        # create a solid from nodes
        solid = self.get_solid_from_nodes(lnodes)
        # A.save_json(filename)

        # find all nodes connected to lnodes not in lnodes
        lneighbors =[]
        for n in lnodes:
            l = self[n].keys()
            lneighbors.extend(l)
        lneighbors = list(set(lneighbors))
        lvalid = [x for x in lneighbors if x not in lnodes]
        # create point cloud from solid
        pcloud = pc.PointCloud()
        pcloud = pcloud.from_solid(solid)
        # get point cloud signature
        pcloud.signature()
        # save assembly in step and json format
        filejson = pcloud.sig + '.json'
        filestep = pcloud.sig + '.stp'
        A.save_json(filejson)
        dirname = self.get_dirname()
        solid.to_step(os.path.join(dirname, filestep))
        # add new assembly node
        new_node = max(self.node.keys()) + 1
        V = np.eye(3)
        ptc = np.array([0, 0, 0])
        self.add_node(new_node,
                      name=pcloud.sig,
                      V=V,
                      pc=ptc,
                      volume=solid.volume(),
                      assembly=True,
                      sig=pcloud.sig)
        self.nnodes += 1
        self.pos[new_node] = solid.center()

        # connect assembly nodes to valid nodes
        for n in lvalid:
            self.add_edge(new_node, n,
                          close=False,
                          intersect=False,
                          sim=False,
                          equal=False)
        # delete nodes from lnodes
        self.remove_nodes(lnodes)

        return A

    def save_gml(self):
        r"""Save Assembly to gml"""
        if not self.isclean:
            self.clean()
            self.isclean = True
        self.serialize()
        filename = self.origin.replace('.stp', '.gml')
        nx.write_gml(self, filename)
        self.unserialize()

    def write_components(self):
        r""" Write individual components of the assembly

        Notes
        -----

        Write unique components to their own step files in a
        subdirectory of the folder containing the original file

        """
        if os.path.isfile(self.origin):
            # directory = os.path.dirname(self.origin)
            # basename = os.path.basename(self.origin)
            # subdirectory = os.path.join(directory,
            #                            os.path.splitext(basename)[0])
            subdirectory = self.get_dirname()
            if not os.path.isdir(subdirectory):
                os.mkdir(subdirectory)
        else:
            msg = "The components of the assembly should already exist"
            raise ValueError(msg)

        # get the list of step files or json files in subdirectory

        filelist = [f for f in os.listdir(subdirectory)
                    if (f.endswith(".stp") or f.endswith(".json"))]

        for f in filelist:
            os.remove(os.path.join(subdirectory, f))

        # creates dataframe
        self.df_nodes = pd.DataFrame(columns=('name', 'count', 'nodes', 'volume', 'assembly'))
        self.dnodes ={}
        for k in self.node:
            # calculate point cloud signature
            # pcloudk = self.node[k]['pcloud']
            #
            # solidk : uncentered solid
            # solidk_centered : centered solid
            #
            solidk = self.node[k]['shape']
            ptc = np.array(solidk.center())

            # warning : center of gravity is obtained
            #           from solid not from pointcloud
            solidk_centered = cm.translated(solidk, -ptc)
            pcloudk = pc.PointCloud()
            pcloudk = pcloudk.from_solid(solidk_centered)
            Npoints = pcloudk.p.shape[0]
            pcloudk.sorting()
            pcloudk.ordering()
            pcloudk.signature()

            V = pcloudk.V

            Npoints = pcloudk.Npoints

            self.node[k]['pc'] = ptc
            self.node[k]['flip'] = False
            self.node[k]['Npoints'] = Npoints

            if np.linalg.det(V)>0:
                self.node[k]['V'] = V
            else:
                V[:, 2] = -V[:, 2]
                self.node[k]['V'] = V
                self.node[k]['flip'] = True
            # V is a assert as a rotation
            assert(np.linalg.det(V) > 0)

            assembly = self.node[k]['assembly']
            #
            # Transfer solidk to the origin
            #
            # The order of the geometrical operations is important
            #
            solidk.translate(-ptc)
            if self.node[k]['flip']:
                solidk.mirrorz()
            solidk.unitary(V.T)
            # assert(np.allclose(solidk.center(),0))
            #
            # Point cloud of the solid centered and transformed
            #
            pcloudk_transformed = pc.PointCloud()
            pcloudk_transformed = pcloudk_transformed.from_solid(solidk)
            pcloudk_transformed.sorting()
            pcloudk_transformed.ordering()
            pcloudk_transformed.signature()
            self.node[k]['sig'] = pcloudk_transformed.sig
            name = pcloudk_transformed.name
            self.node[k]['name'] = name
            self.node[k]['pcloud'] = pcloudk_transformed

            filename = pcloudk_transformed.name + ".stp"
            filename = os.path.join(subdirectory, filename)
            lnames = self.df_nodes['name'].values
            # if not os.path.isfile(filename):
            if not (name in lnames):
                # save translated transformed unique shape to filename
                solidk.to_step(filename)
                index = len(self.df_nodes)
                self.df_nodes = self.df_nodes.set_value(index, 'name', name)
                self.df_nodes = self.df_nodes.set_value(index,
                                                        'volume',
                                                        solidk.volume())
                self.df_nodes = self.df_nodes.set_value(index, 'count', 1)
                self.df_nodes = self.df_nodes.set_value(index, 'nodes', [k])
                self.df_nodes = self.df_nodes.set_value(index,
                                                        'assembly',
                                                        assembly)
            else:
                # get solid from origin file
                solid_orig = cm.from_step(filename)
                # transform it around origin
                # node_solid = self.df_nodes[self.df_nodes['name']==name]['nodes'].values[0][0]
                # print("Node_solid",k,node_solid)
                # solid.unitary(V_orig)
                # pcloud_orig = pc.PointCloud()
                # pcloud_orig = pcloud_orig.from_solid(solid_orig)
                # pcloud_orig.sorting()
                # pcloud_orig.ordering()
                # d0,d1 = pcloud_orig.distance(pcloudk_transformed)
                # S = np.dot(V.T,V_orig)
                # T1 = np.dot(pcloud_orig.p.T,pcloudk_transformed.p)
                # T2 = np.dot(pcloudk_transformed.p.T,pcloud_orig.p)
                # print(k, d0, d1)
                # SI = np.diag(1./Sk)
                # U3 = Uk[:,:3]
                # H1 = np.dot(pcloud_orig.p.T,U3)
                # H2 = np.dot(H1,SI)
                # T3 = np.dot(H2,Vk)
                # if k==7:
                #     pdb.set_trace()
                # self.node[k]['V'] = V_orig
                # print(d0,d1)
                # if pcloudk != pcloud_orig:
                #     V1 = pcloud_orig.get_transform(pcloudk)
                #     V2 = pcloudk.get_transform(pcloud_orig)
                #     fig,ax = pcloudk.show(c = 'r')
                #     fig,ax = pcloud_orig.show(fig=fig, ax=ax, c='b')
                #     plt.show()
                #     pdb.set_trace()

                dfname = self.df_nodes[self.df_nodes['name'] == name]
                dfname['count'] += 1
                dfname.iloc[0]['nodes'].append(k)
                self.df_nodes[self.df_nodes['name'] == name] = dfname
            self.dnodes[k] = index

    def get_dirname(self):
        """ get dirname from self.origin

        Notes
        -----

        If the origin file is a step file, there is a creation of
        directory with the same name where all the derivated files
        will be placed, including step files and json files.

        """
        ext = os.path.splitext(self.origin)[1]
        if ext == '.stp' or ext == '.step':
            dirname = os.path.splitext(self.origin)[0]
        else:
            dirname = os.path.dirname(self.origin)
        return dirname

    def get_solid_from_nodes(self, lnodes):
        """ get a solid from nodes

        Parameters
        ----------
        lnodes : list of nodes or -1 for all

        Returns
        -------
        s : node solid

        Notes
        -----

        If the assembly file has extension .stp it means that the analysis has
        not been done. After the analysis a directory has been created, it
        contains all the .stp file of the parts and a .json file which contains
        the graph information.

        """
        if lnodes == -1:
            lnodes = self.node.keys()

        rep = self.get_dirname()

        lfiles = [str(self.node[k]['name'])+'.stp' for k in lnodes]
        lV = [self.node[k]['V'] for k in lnodes]
        lflip = [self.node[k]['flip'] for k in lnodes]
        lpc = [self.node[k]['pc'] for k in lnodes]
        solid = cm.Solid([])

        for k, s in enumerate(lfiles):
            filename = os.path.join(rep, s)
            shp = cm.from_step(filename)
            # assert(np.allclose(np.array(shp.center()),0)),pdb.set_trace()
            V = lV[k]
            shp.unitary(V)
            if lflip[k]:
                shp.mirrorz()
            shp.translate(lpc[k])
            solid = solid + shp
            # print(k,solid.shape.Orientation(),solid.volume(),shp.volume())

        return solid

    def view(self, node_index=-1, jupyter=False):
        """ view assembly

        Parameters
        ----------

        node_index : a list of Assembly nodes (-1 : all nodes)

        Notes
        -----

        An Assembly is a graph
        Each node of an assembly has attached
            + a filename describing a solid in its own local frame
            + a translation vector for solid placement in the global frame

        This function produces the view of the assembly in the global frame.

        """
        if type(node_index) == int:
            if node_index == -1:
                node_index = self.node.keys()
            else:
                node_index=[node_index]

        assert(max(node_index) <= max(self.node.keys())), "Wrong node index"

        if self.serialized:
            s.unserialize()

        solid = self.get_solid_from_nodes(node_index)

        # solid.to_html('assembly.html')
        if jupyter:
            j = jupyter_renderer.JupyterRenderer()
            j.DisplayShape(solid.shape, update=True)
            return solid, j
        else:
            my_renderer = threejs_renderer.ThreejsRenderer()
            my_renderer.DisplayShape(solid.shape)
            my_renderer.render()

            return solid


def reverse(step_filename, view=False):
    r"""Reverse STEP file using ccad

    Parameters
    ----------
    step_filename : str
        Path to the STEP file
    view : bool, optional (default is False)
        Launch the ccad viewer?

    """
    # read a step file and add nodes to graph
    assembly = Assembly()
    assembly.from_step(step_filename)
    # write a separate step file for each node
    # print("write_components")
    # tic = time.time()
    assembly.write_components()
    # toc = time.time()
    # print(toc-tic)
    # tag and analyze nodes - creates edges between nodes based
    # on dicovered pointcloud similarity and proximity
    #
    # similarity precursor of symmetry
    # proximity precursor of contact
    # join axiality precursor of co-axiality (alignment)
    #
    print("equal_sim_nodes_edges")
    assembly.equalsim_nodes_edges()
    print("delete_edges")
    # assembly.delete_edges(kind='equal')
    print("intersect_nodes_edges")
    assembly.intersect_nodes_edges()
    # assembly saving
    # assembly.save_gml()
    print('save json')
    # assembly.save_json()

    if view:
        ccad_viewer = cd.view()
        for shell in assembly.shape.subshapes("Shell"):
            ccad_viewer.display(shell)
        cd.start()

    return assembly


# def view(step_filename):
#     r"""View the STEP file contents in the aocutils wx viewer.
#
#     Parameters
#     ----------
#     step_filename : str
#         path to the STEP file
#
#     """
#
#     importer = StepImporter(filename=step_filename)
#
#     class MyFrame(wx.Frame):
#         r"""Frame for testing"""
#         def __init__(self):
#             wx.Frame.__init__(self, None, -1)
#             self.p = Wx3dViewer(self)
#             for shape in importer.shapes:
#                 self.p.display_shape(shape)
#             self.Show()
#
#     app = wx.App()
#     frame = MyFrame()
#     app.SetTopWindow(frame)
#     app.MainLoop()

def intersect(s1, s2):
    """ Determine intersection of 2 shapes bounding boxes

    Parameters
    ----------
    s1 : solid 1
    s2 : solid 2

    Returns
    -------
    boolean

    """
    bb1 = s1.bounding_box()
    bb2 = s2.bounding_box()

    Intx1 = interval([bb1[0, 0], bb1[1, 0]])
    Inty1 = interval([bb1[0, 1], bb1[1, 1]])
    Intz1 = interval([bb1[0, 2], bb1[1, 2]])

    Intx2 = interval([bb2[0, 0], bb2[1, 0]])
    Inty2 = interval([bb2[0, 1], bb2[1, 1]])
    Intz2 = interval([bb2[0, 2], bb2[1, 2]])

    bx = len(Intx1 & Intx2) > 0
    by = len(Inty1 & Inty2) > 0
    bz = len(Intz1 & Intz2) > 0

    gapx = 0
    gapy = 0
    gapz = 0

    if not bx:
        Ix = (Intx1 | Intx2)
        gapx = Ix[1][0] - Ix[0][1]
    if not by:
        Iy = (Inty1 | Inty2)
        gapy = Iy[1][0] - Iy[0][1]
    if not bz:
        Iz = (Intz1 | Intz2)
        gapz = Iz[1][0] - Iz[0][1]

    bint = np.array([bx, by, bz])
    gapint = np.array([gapx, gapy, gapz])
    return bint, gapint
