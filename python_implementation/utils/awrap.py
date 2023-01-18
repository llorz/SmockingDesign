from typing import *
import time
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as spla

def build_cotan_laplacian( points: np.ndarray, tris: np.ndarray ):
    a,b,c = (tris[:,0],tris[:,1],tris[:,2])
    A = np.take( points, a, axis=1 )
    B = np.take( points, b, axis=1 )
    C = np.take( points, c, axis=1 )

    eab,ebc,eca = (B-A, C-B, A-C)
    eab = eab/np.linalg.norm(eab,axis=0)[None,:]
    ebc = ebc/np.linalg.norm(ebc,axis=0)[None,:]
    eca = eca/np.linalg.norm(eca,axis=0)[None,:]

    alpha = np.arccos( -np.sum(eca*eab,axis=0) )
    beta  = np.arccos( -np.sum(eab*ebc,axis=0) )
    gamma = np.arccos( -np.sum(ebc*eca,axis=0) )

    wab,wbc,wca = ( 1.0/np.tan(gamma), 1.0/np.tan(alpha), 1.0/np.tan(beta) )
    rows = np.concatenate((   a,   b,   a,   b,   b,   c,   b,   c,   c,   a,   c,   a ), axis=0 )
    cols = np.concatenate((   a,   b,   b,   a,   b,   c,   c,   b,   c,   a,   a,   c ), axis=0 )
    vals = np.concatenate(( wab, wab,-wab,-wab, wbc, wbc,-wbc,-wbc, wca, wca,-wca, -wca), axis=0 )
    L = sparse.csc_matrix((vals,(rows,cols)),shape=(points.shape[1],points.shape[1]), dtype=float)
    return L

def build_weights_and_adjacency( points: np.ndarray, tris: np.ndarray, L: Optional[sparse.csc_matrix]=None ):
    L = L if L is not None else build_cotan_laplacian( points, tris )
    n_pnts, n_nbrs = (points.shape[1], L.getnnz(axis=0).max()-1)
    nbrs = np.ones((n_pnts,n_nbrs),dtype=int)*np.arange(n_pnts,dtype=int)[:,None]
    wgts = np.zeros((n_pnts,n_nbrs),dtype=float)

    for idx,col in enumerate(L):
        msk = col.indices != idx
        indices = col.indices[msk]
        values  = col.data[msk]
        nbrs[idx,:len(indices)] = indices
        wgts[idx,:len(indices)] = -values

    return nbrs, wgts, L

class ARAP:
    def __init__( self, points: np.ndarray, tris: np.ndarray, constraint_weights: Dict[int, float],
    constraints: Dict[int, Tuple[float, float, float]]):
        self._pnts    = points.copy()
        self._tris    = tris.copy()
        self._nbrs, self._wgts, self._L = build_weights_and_adjacency( self._pnts, self._tris )

        self._anchors = list(constraint_weights.keys())
        # Weight matrix.
        E = sparse.dok_matrix((self.n_pnts,self.n_pnts),dtype=float)
        for i, weight in constraint_weights.items():
            E[i,i] = np.sqrt(weight)
        E = E.tocsc()
        anchors_weights = E.T@E
        # Right hand side of the constraints.
        f = np.zeros((self.n_pnts,self.n_dims),dtype=float)
        for i, v in constraints.items():
          f[i,:] = v
        self._cos_rhs = anchors_weights * f
        self._constraints = constraints
        self._solver = spla.factorized( ( self._L.T@self._L + anchors_weights).tocsc() )

    def get_points(self):
      return self._pnts

    @property
    def n_pnts( self ):
        return self._pnts.shape[1]

    @property
    def n_dims( self ):
        return self._pnts.shape[0]

    @property
    def anchors( self ):
      return self._anchors

    def __call__( self, anchors: Dict[int,Tuple[float,float,float]], num_iters: Optional[int]=4, def_points = None ):
        R = np.array([np.eye(self.n_dims) for _ in range(self.n_pnts)])
        if def_points is None:
          def_points = self._solver( self._L.T@self._build_rhs(R) + self._cos_rhs )
          for i, v in self._constraints.items():
            def_points[i,:] = v
        for i in range(num_iters):
            R = self._estimate_rotations( def_points.T )
            def_points = self._solver( self._L.T@self._build_rhs(R) + self._cos_rhs )
        return def_points.T

    def _estimate_rotations( self, def_pnts: np.ndarray ):
        tru_hood = (np.take( self._pnts, self._nbrs, axis=1 ).transpose((1,0,2)) - self._pnts.T[...,None])*self._wgts[:,None,:]
        rot_hood = (np.take( def_pnts,   self._nbrs, axis=1 ).transpose((1,0,2)) - def_pnts.T[...,None])

        U,s,Vt = np.linalg.svd( rot_hood@tru_hood.transpose((0,2,1)) )
        R = U@Vt
        dets = np.linalg.det(R)
        Vt[:,self.n_dims-1,:] *= dets[:,None]
        R = U@Vt
        return R

    def _build_rhs( self, rotations: np.ndarray ):
        R = (np.take( rotations, self._nbrs, axis=0 )+rotations[:,None])*0.5
        tru_hood = (self._pnts.T[...,None]-np.take( self._pnts, self._nbrs, axis=1 ).transpose((1,0,2)))*self._wgts[:,None,:]
        rhs = np.sum( (R@tru_hood.transpose((0,2,1))[...,None]).squeeze(), axis=1 )
        return rhs