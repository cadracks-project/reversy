import numpy as np
import scipy.spatial as spa
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os
import sys
import json
import pdb

def getname(**kwargs):
    """ naming a parts

    Parameters
    ----------

    sector(1) : str
        (CAR, AERO, GENERIC, MECA, ELEC)
    domain(2)  : str
        (SUSPENSION,WING,FASTENER)
    function(3) :
        (PISTON,SPAR,SCREW)
    dimension(4) (real separated by #)
        (21#29,7)
    material(5) (STEEL,CARBON, PVC)
    origin(6) (GENERIC, AUDI)
    alpha(7) (arbitrary number, external ref)

    Examples
    --------

    >>> name = getname(material="PAPER")

    """

    sector = kwargs.pop('sector','')
    domain = kwargs.pop('domain','')
    function = kwargs.pop('function','')
    dimension = kwargs.pop('dimension','')
    material = kwargs.pop('material','')
    origin = kwargs.pop('origin','')
    alpha = kwargs.pop('alpha','1')

    lattributes = [sector,domain,function,dimension,material,origin,alpha]
    name = "_".join(lattributes)

    return(name)

class PointCloud(object):
    def __init__(self,p=np.array([[]]),ndim=3):
        """
        Parameters
        ----------

        p : np.array (Npoints x ndim)
        ndim : int
            default : 3

        pc : point cloud centroid
        p : centered ordered point cloud
        dist : ordered distance

        """
        self.ndim = ndim
        if p.size==0:
            p.shape = (0,ndim)
        assert(p.shape[1]==ndim)
        self.Npoints = p.shape[0]
        self.p = p
        self.centered = False
        self.ordered = False

    def __add__(self,p):
        P = PointCloud()
        if p.shape[1] != self.ndim:
            P.p = np.vstack((self.p,p.T))
        else:
            P.p = np.vstack((self.p,p))
        P.Npoints = P.p.shape[0]
        return(P)

    def __repr__(self):
        st = 'PointCloud : '+str(self.Npoints)+' points\n'
        if self.centered:
            st = st + 'centered ' + '\n'
        if self.ordered:
            st = st + 'ordered ' + '\n'

        if hasattr(self,'sig'):
            st = st + 'Signature : ' + self.sig + '\n'
        if hasattr(self,'pc'):
            st = st + 'pc : ' + str(self.pc) + '\n'
        if hasattr(self,'V'):
            st = st + 'V : ' + str(self.V) + '\n'
        return st

    def from_solid(self,solid):
        vertices = solid.subshapes("Vertex")
        for vertex in vertices:
            point = np.array(vertex.center())[:, None]
            self = self + point
        return self

    def mindist(self,other):
        """

        Parameters
        ----------

        other : point cloud

        """

        min_x = np.min(self.p[:,0])
        min_y = np.min(self.p[:,1])
        min_z = np.min(self.p[:,2])
        mino_x = np.min(other.p[:,0])
        mino_y = np.min(other.p[:,1])
        mino_z = np.min(other.p[:,2])

        max_x = np.max(self.p[:,0])
        max_y = np.max(self.p[:,1])
        max_z = np.max(self.p[:,2])
        maxo_x = np.max(other.p[:,0])
        maxo_y = np.max(other.p[:,1])
        maxo_z = np.max(other.p[:,2])

        dx1 = mino_x - max_x
        dx2 = min_x - maxo_x

        dy1 = mino_y - max_y
        dy2 = min_y - maxo_y

        dz1 = mino_z - max_z
        dz2 = min_z - maxo_z

    def center(self):
        if not self.centered:
            self.pc = np.mean(self.p, axis=0)
            return self.pc

    def centering(self):
        #
        # centering
        #
        self.pc = np.mean(self.p, axis=0)
        self.p = self.p - self.pc
        assert(self.p.shape[0]==self.Npoints)
        self.centered = True

    def ordering(self):
        #
        # sorting points w.r.t distance to origin
        # This ordering is needed for PointClouds comparison
        #
        d = np.sqrt(np.sum(self.p*self.p,axis=1))
        self.u = np.argsort(d)
        self.dist = d[self.u]
        self.p = self.p[self.u,:]
        self.ordered = True
        assert(self.p.shape[0]==self.Npoints), pdb.set_trace()


    def signature(self):
        r""" Signature of a point cloud using SVD

        sig : str
        V   :
        pc : middle point / centroid of the point cloud
        vec
        ang

        """

        if not(self.centered):
            self.centering()

        #self.ordering()

        U, S, V = np.linalg.svd(self.p)

        minx = np.min(self.p[:,0])
        miny = np.min(self.p[:,1])
        minz = np.min(self.p[:,2])
        maxx = np.max(self.p[:,0])
        maxy = np.max(self.p[:,1])
        maxz = np.max(self.p[:,2])
        bbc = np.array([maxx-minx,maxy-miny,maxz-minz])

        B0 = str(int(np.ceil(bbc[0])))
        B1 = str(int(np.ceil(bbc[1])))
        B2 = str(int(np.ceil(bbc[2])))

        S0 = str(int(np.ceil(S[0])))
        S1 = str(int(np.ceil(S[1])))
        if S[2]<1e-12:
            S2 = '0'
            sig = S0 + '#' + S1+ "#" + S2 + '_' + B0 + "#" + B1 + '#' + B2
            name = getname(dimension=sig,function='SYMAX')
        else:
            S2 = str(int(np.ceil(S[2])))
            sig = S0 + '#' + S1+ "#" + S2 + '_' + B0 + "#" + B1 + '#' + B2
            if S2=='1':
                name = getname(dimension=sig,function='ALMSYM')
            else:
                name = getname(dimension=sig)

        self.sig = sig
        self.name = name
        self.bbc = bbc
        self.V = V

    def show(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        c = 'b'
        m = 'o' # '^'
        ax.scatter(self.p[:,0], self.p[:,1], self.p[:,2], c=c, marker=m)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()
        return fig,ax

