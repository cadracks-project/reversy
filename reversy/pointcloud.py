import numpy as np
import scipy.spatial as spa
import quaternions as cq
import pdb

class PointCloud(object):
    def __init__(self,p=np.array([[]]),ndim=3):

        self.p = p
        self.ndim = ndim
        if p.size==0:
            self.p.shape = (0,ndim)
        assert(self.p.shape[1]==ndim)
        self.N = self.p.shape[0]

    def __add__(self,p):
        P = PointCloud()
        if p.shape[1]!=self.ndim:
            P.p = np.vstack((self.p,p.T))
        else:
            P.p = np.vstack((self.p,p))
        P.N = P.p.shape[0]
        return(P)

    def __repr__(self):
        st = 'PointCloud : '+str(self.N)+' points\n'
        if hasattr(self,'sig'):
            st = st + 'Signature : ' + self.sig + '\n'
        if hasattr(self,'ptm'):
            st = st + 'ptm : ' + str(self.ptm) + '\n'
        if hasattr(self,'vec'):
            st = st + 'vec : ' + str(self.vec) + '\n'
        if hasattr(self,'ang'):
            st = st + 'ang : ' + str(self.ang) + '\n'
        return(st)

    def signature(self):
        r"""Signature of a point cloud using Singular Values Decomposition (SVD)

        sig : str
        V   :
        ptm : middle point / barycentre of the point cloud
        q : quaternion from V
        vec
        ang

        """
        #pts = np.vstack((self.p[0, :], self.p[1, :], self.p[2, :])).T
        # pts : N x dim
        pts = self.p
        #logger.debug("Shape of pts : %s" % str(pts.shape))
        ptm = np.mean(pts, axis=0)
        #logger.debug("Mean pt : %s" % str(ptm))
        ptsm = pts - ptm
        #logger.debug("Shape of ptsm : %s" % str(ptsm.shape))  # should be as pts
        U, S, V = np.linalg.svd(ptsm)
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
        S2 = str(int(np.ceil(S[2])))

        sig = S0 + "_" + S1 + "_" + S2

        self.sig = sig
        self.V = V
        # gravity center
        self.ptm =ptm
        # q : quaternion from V
        self.q = q
        # vec :  rotation axis
        self.vec = vec
        self.ang = ang
