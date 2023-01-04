# Shamefully stolen from https://gist.github.com/jamesgregson/ba72af28c83b3da1968690cd11278829
from typing import *
import time
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
import bpy
import itertools
from mathutils import Vector
import bmesh
import arap_stitching_lines

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
    L = sparse.coo_matrix((vals,(rows,cols)),shape=(points.shape[1],points.shape[1]), dtype=float).tocsc()
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

class SmockARAP:
    def __init__( self, points: np.ndarray, tris: np.ndarray, lines, anchor_weight: Optional[float]=10.0, L: Optional[sparse.csc_matrix]=None ):
        self._pnts    = points.copy()
        self._tris    = tris.copy()
        self._nbrs, self._wgts, self._L = build_weights_and_adjacency( self._pnts, self._tris, L )

        self._anc_wgt = anchor_weight
        E = self._get_constraints(lines)
        Ez = self._get_z_constraints(lines)
        self._Emaxembed, inds = self._get_max_embed_constraints()
        self._corners = points[:, list(inds)].T
        self._Ez = Ez
        self._E = E
        self._ncnstr = E.shape[0]
        # first_row = sparse.hstack([self._L.T@self._L, self._anc_wgt*E.T])
        # second_row = sparse.hstack([self._anc_wgt * E, sparse.csr_matrix((self._ncnstr, self._ncnstr))])
        # A = sparse.vstack([first_row, second_row])
        # self._solver = spla.factorized( ( A).tocsc() )
        # first_row2 = sparse.hstack([self._L.T@self._L, self._anc_wgt*Ez.T])
        # second_row2 = sparse.hstack([self._anc_wgt * Ez, sparse.csr_matrix((Ez.shape[0], Ez.shape[0]))])
        # A2 = sparse.vstack([first_row2, second_row2])
        # self._solver2 = spla.factorized( ( A2).tocsc() )
        self._solver = spla.factorized( ( self._L.T@self._L +
          self._anc_wgt*E.T@E))
        self._solver2 = spla.factorized( ( self._L.T@self._L + self._anc_wgt*Ez.T@Ez).tocsc() )

    @property
    def n_pnts( self ):
        return self._pnts.shape[1]

    @property
    def n_dims( self ):
        return self._pnts.shape[0]

    def __call__( self, unused, num_iters: Optional[int]=4, def_points = None ):
        # con_rhs = self._E.T @ self._build_constraint_rhs()
        R = np.array([np.eye(self.n_dims) for _ in range(self.n_pnts)])
        zeros = np.zeros([self._E.shape[0], self.n_dims])
        if def_points is None:
          # def_points = self._pnts.T
          rhs = self._build_rhs(R)
          # def_points = self._solver(np.concatenate([rhs, zeros], axis=0))
          # def_points = def_points[:self.n_pnts, :]

          r1 = self._solver( self._L.T@rhs[:, 0:2] )
          r2 = np.reshape(self._solver2( self._L.T@rhs[:, 2] ), [self.n_pnts, 1])
          def_points = np.concatenate([r1, r2], axis=1)
          # def_points = self._solver(self._L.T@rhs)
        for i in range(num_iters):
            R = self._estimate_rotations( def_points.T )
            rhs = self._build_rhs(R)
            # def_points = self._solver(self._L.T@rhs)
            # r1 = self._solver(np.concatenate([self._L.T@rhs[:, 0:2], zeros[:, 0:2]], axis=0))
            # r2 = self._solver2(np.concatenate([
            #   np.reshape(self._L.T@rhs[:, 2], [rhs.shape[0], 1]), 
            #   np.zeros([self._Ez.shape[0], 1])], axis=0))
            # r1 = r1[:self.n_pnts, :]
            # r2 = r2[:self.n_pnts, :]
            # def_points = np.concatenate([r1, r2], axis=1)

            # def_points = self._solver(np.concatenate([self._L.T@rhs, zeros], axis=0))
            # def_points = def_points[:self.n_pnts, :]
            # rhs = self._build_rhs(R)
            r1 = self._solver( self._L.T@rhs[:, 0:2] )
            r2 = np.reshape(self._solver2( self._L.T@rhs[:, 2] ), [self.n_pnts, 1])
            def_points = np.concatenate([r1, r2], axis=1)
        return def_points.T

    def _get_max_embed_constraints(self):
      nnz = self._L.getnnz(axis=0)-1
      inds = np.where(nnz < 4)[0]
      E = sparse.dok_matrix((len(inds), self.n_pnts),dtype=float)
      for i in range(len(inds)):
        E[i, inds[i]] = 1.0
      return E.tocsc(), inds

    def _get_constraints(self, lines):
      # num_stitching_verts = np.sum([len(x) for x in lines])
      num_stitching_pairs = len(lines)
      E = sparse.dok_matrix((num_stitching_pairs, self.n_pnts),dtype=float)
      cindex = 0
      for l in lines:
        E[cindex, l[0]] = 1.0
        E[cindex, l[1]] = -1.0
        cindex += 1
      return E.tocsc()

    def _get_z_constraints(self, lines):
      # num_stitching_verts = np.sum([len(x) for x in lines])
      indices = {i for x in lines for i in x}
      num_stitching_verts = len(indices)
      E = sparse.dok_matrix((num_stitching_verts, self.n_pnts),dtype=float)
      cindex = 0
      for ind in indices:
        E[cindex, ind] = 1.0
        cindex += 1
      return E.tocsc()

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

    def _build_constraint_rhs( self ):
        f = np.zeros((self._ncnstr,self.n_dims),dtype=float)
        return f


def extract_graph_from_meshgrid(gx, gy):
    """extract the mesh/graph from the grid"""
    # number of vertices along x/y directions 
    numx = len(gx)
    numy = len(gx[0])
    
    num = numx * numy
    
    # index of the points
    ind = np.array(range(num)).reshape(numx, numy)
    
    # create the face from the four corners
    top_left = ind[:numx-1,:numy-1]
    bottom_left = ind[1:numx, :numy-1]
    top_right = ind[:numx-1, 1:numy]
    bottom_right = ind[1:numx, 1:numy]
    
    # create the grid vertices: list of array - otherwise blender gets confused :/
    F = np.array([top_left.flatten().tolist(),
    top_right.flatten().tolist(),
    bottom_right.flatten().tolist(),
    bottom_left.flatten().tolist()])
    F = list(F.transpose()) #list of array
    
    # create the grid vertices: array
    V = np.array([gx.flatten().tolist(), gy.flatten().tolist()]).transpose()
    
    return F, V

def get_fine_grid(fsp):
  bbox_min, bbox_max = (np.amin(fsp.V, axis=0), np.amax(fsp.V, axis=0))
  margin_x, margin_y = (0.5, 0.5)
  grid_size = [int((bbox_max[0] - bbox_min[0] + 2*margin_x) / 0.15),
               int((bbox_max[1] - bbox_min[1] + 2*margin_y) / 0.15)]
  # Create grid.
  [gx, gy]= np.meshgrid( \
    np.linspace(bbox_min[0] - margin_x, bbox_max[0] + margin_x, grid_size[0]),
    np.linspace(bbox_min[1] - margin_y, bbox_max[1] + margin_y, grid_size[1]))
  
  # Create graph from grid. 
  F, V = extract_graph_from_meshgrid(gx, gy)
  return F, V

def get_stitching_lines_in_new_grid(fsp, vid):
  nl = fsp.num_stitching_lines()
  edges = []
  for lid in range(nl):
    stitching_vertices = fsp.get_vid_in_stitching_line(lid)
    for i in range(len(stitching_vertices) - 1):
      # Edge from stitching_vert i -> i + 1 in the new grid.
      edges.append((vid[stitching_vertices[i]], vid[stitching_vertices[i + 1]]))
  return edges

def create_direct_arap_mesh(fsp, sg):
  F, V = get_fine_grid(fsp)
  F = np.array(list(itertools.chain.from_iterable(\
      [(f[[0,1,2]], f[[2,3,0]]) for f in F])))
  # Correspondence to vertices in the coarse grid.
  vid = [np.argmin(np.linalg.norm(V - p, axis=1)) for p in fsp.V]
  # Stitching lines constraints in the fine grid.
  stitching_edges = get_stitching_lines_in_new_grid(fsp, vid)
  V3D = np.concatenate((V, 1e-4 * np.random.rand(V.shape[0], 1)), axis=1)
  smock_arap = arap_stitching_lines.SmockARAP(V3D.transpose(), F, stitching_edges, 0.004)
  # Create mesh.
  mesh = bpy.data.meshes.new("cloth_sim_mesh")
  mesh.from_pydata(V3D, [], F)
  # Move non-underlay nodes up.
  indices = {i for x in stitching_edges for i in x}
  for i in range(V3D.shape[0]):
    if i in indices:
      continue
    V3D[i, 2] = 0.5
  # V3D = smock_arap(None, 0, V3D.T)
  mesh.vertices.foreach_set("co", V3D.reshape(V3D.size))
  mesh.update()

  # Add uv
  uvlayer = mesh.uv_layers.new()
  for face in mesh.polygons:
    for vert_idx, loop_idx in zip(face.vertices, face.loop_indices):
        uvlayer.data[loop_idx].uv = (V[vert_idx, 0] / 20.0, V[vert_idx, 1] / 20.0)
  
  # Create object and collection
  obj = bpy.data.objects.new('cloth_sim_obj', mesh)
  obj.location = (0, 0, 0)
  obj.data.materials.append(bpy.data.materials['Fabric035'])
  for f in obj.data.polygons:
    f.use_smooth = True
  smocked_collection = bpy.data.collections.new('cloth_sim_collection')
  bpy.context.scene.collection.children.link(smocked_collection)
  smocked_collection.objects.link(obj)

  # Add stitching edges.
  
  bpy.context.view_layer.objects.active = obj
  bpy.ops.object.mode_set(mode='EDIT')
  bm = bmesh.from_edit_mesh(obj.data)
  bm.verts.ensure_lookup_table() # Required?
  for e in stitching_edges:
    bm.edges.new((bm.verts[e[0]], bm.verts[e[1]]))
  bpy.ops.object.mode_set(mode='OBJECT')

  # Create arap.


  return obj, V3D, F, smock_arap