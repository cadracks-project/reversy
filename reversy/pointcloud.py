import numpy as np
import scipy.spatial as spa
import quaternions as cq
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

    def __add__(self,p):
        P = PointCloud()
        if p.shape[1]!=self.ndim:
            P.p = np.vstack((self.p,p.T))
        else:
            P.p = np.vstack((self.p,p))
        P.N = P.p.shape[0]
        return(P)

    def __repr__(self):
        st = 'PointCloud : '+str(self.Npoints)+' points\n'
        if hasattr(self,'sig'):
            st = st + 'Signature : ' + self.sig + '\n'
        if hasattr(self,'pc'):
            st = st + 'pc : ' + str(self.pc) + '\n'
        if hasattr(self,'vec'):
            st = st + 'vec : ' + str(self.vec) + '\n'
        if hasattr(self,'ang'):
            st = st + 'ang : ' + str(self.ang) + '\n'
        return(st)

    def mindist(self,otherpc):
        min_x = np.min(self.p[:,0])
        min_x = np.min(self.p[:,0])
        min_x = np.min(self.p[:,0])
        min_x = np.min(self.p[:,0])

    def centering(self):
        #
        # centering
        #
        self.pc = np.mean(self.p, axis=0)
        self.p= self.p - self.pc

    def ordering(self):
        #
        # sorting points w.r.t distance to origin
        # This ordering is needed for PointClouds comparison
        #
        d = np.sqrt(np.sum(self.p*self.p,axis=0))
        self.u = np.argsort(d)
        self.dist = d[self.u]
        self.p = self.p[self.u,:]

    def signature(self):
        r""" Signature of a point cloud using SVD

        sig : str
        V   :
        pc : middle point / centroid of the point cloud
        q : quaternion from V
        vec
        ang

        """

        self.centering()

        self.ordering()

        U, S, V = np.linalg.svd(self.p)
        #logger.debug("U shape : %s" % str(U.shape))  # rotation matrix (nb_pts x nb_pts)
        #logger.debug("S shape : %s" % str(S.shape))  # Diagonal matrix (3d vec)
        #logger.debug(str(S))
        #logger.debug("V shape : %s" % str(V.shape))  # rotation matrix (3x3)
        #logger.debug(str(V))

        q = cq.Quaternion()
        q.from_mat(V)
        vec, ang = q.vecang()
        #logger.debug("Vec : %s" % str(vec))
        #logger.debug("Ang : %f" % ang)

        S0 = str(int(np.ceil(S[0])))
        S1 = str(int(np.ceil(S[1])))
        if S[2]<1e-6:
            S2 = '0'
            name = getname(dimension=S0+'#'+S1+'#'+S2,function='CYLINDER')
        else:
            S2 = str(int(np.ceil(S[2])))
            name = getname(dimension=S0+'#'+S1+'#'+S2)

        sig = S0 + "_" + S1 + "_" + S2

        self.sig = sig
        self.name = name
        self.V = V
        # gravity center
        # q : quaternion from V
        self.q = q
        # vec :  rotation axis
        self.vec = vec
        self.ang = ang


