#!/usr/bin/python
# coding: utf-8
"""
Decomposing an assembly obtained from a STEP file

"""

from __future__ import print_function
import wx
import os
import logging
from os import path as _path
import ccad.model as cm
import ccad.display as cd
from aocxchange.step import StepImporter
from aocutils.display.wx_viewer import Wx3dViewer
import numpy as np
import networkx as nx
import ccad.model as ccm
import pointcloud as pc
logger = logging.getLogger('__name__')

class Assembly(object):
    r""" Assembly Class

    This class has to be connected to the osvcad Assembly representation

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
            If True, directly use the point cloud of the Shell
            If False, iterate the faces, wires and then vertices

        """

        self.shape = shape
        self.G = nx.DiGraph()
        self.G.pos = dict()
        self.origin = origin

        shells = self.shape.subshapes("Shell")
        logger.info("%i shells in assembly" % len(shells))

        for k, shell in enumerate(shells):
            logger.info("Dealing with shell nb %i" % k)
            self.G.pos[k] = shell.center()
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

            self.G.add_node(k, pcloud=pcloud, shape=shell)

    def __repr__(self):
        st = self.shape.__repr__()+'\n'
        st += self.G.__repr__()+'\n'
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
            If False, iterate the faces, wires and then vertices

        Returns
        -------
        Assembly : the new Assembly object created from a STEP file

        """
        solid = cm.from_step(filename)
        return cls(solid, origin=filename, direct=direct)

    def tag_nodes(self):
        r"""Add computed data to each node of the assembly"""

        for k in self.G.node:
            # sig, V, ptm, q, vec, ang = signature(self.G.node[k]['pcloud'])
            self.G.node[k]['pcloud'].signature()
            #self.G.node[k]['name'] = sig
            #self.G.node[k]['R'] = V
            #self.G.node[k]['ptm'] = ptm
            #self.G.node[k]['q'] = q

    def write_components(self):
        r"""Write components of the assembly to their own step files in a
        subdirectory of the folder containing the original file"""
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

        for k in self.G.node:
            #sig, V, ptm, q, vec, ang = signature(self.G.node[k]['pcloud'])
            self.G.node[k]['pcloud'].signature()

            sig = self.G.node[k]['pcloud'].sig
            vec = self.G.node[k]['pcloud'].vec
            ang = self.G.node[k]['pcloud'].ang
            ptm = self.G.node[k]['pcloud'].ptm

            shp = self.G.node[k]['shape']
            filename = sig + ".stp"
            if not os.path.isfile(filename):
                shp.translate(-ptm)
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

    assembly = Assembly.from_step(step_filename, direct=False)
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
    filename = "../step/ASM0001_ASM_1_ASM.stp"  # OCC compound
    # filename = "../step/MOTORIDUTTORE_ASM.stp" # OCC compound
    #filename = "../step/aube_pleine.stp"  # OCC Solid

    a1 = reverse(filename)
    cd.view(a1)
