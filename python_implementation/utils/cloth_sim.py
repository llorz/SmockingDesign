
import bpy
from bpy.types import Operator
import numpy as np
from mathutils import Vector
import bmesh
import itertools

LAYOUT_TEXT = 0.4

def extract_graph_from_meshgrid(gx, gy, if_add_diag=True):
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
  F, V = extract_graph_from_meshgrid(gx, gy, True)
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

def create_cloth_mesh(fsp, add_text_to_scene):
  F, V = get_fine_grid(fsp)
  F = np.array(list(itertools.chain.from_iterable(\
      [(f[[0,1,2]], f[[2,3,0]]) for f in F])))
  # Create mesh.
  V3D = np.concatenate((V, np.zeros([V.shape[0], 1])), axis=1)
  mesh = bpy.data.meshes.new("cloth_sim_mesh")
  mesh.from_pydata(V3D, [], F)
  mesh.update()

  # Add uv
  uvlayer = mesh.uv_layers.new()
  for face in mesh.polygons:
    for vert_idx, loop_idx in zip(face.vertices, face.loop_indices):
        uvlayer.data[loop_idx].uv = (V[vert_idx, 0] / 20.0, V[vert_idx, 1] / 20.0)
  
  # Create object and collection
  obj = bpy.data.objects.new('cloth_sim_obj', mesh)
  obj.location = (np.max(fsp.V[:, 0]), np.max(fsp.V[:, 1]) + 4, 0)
  obj.data.materials.append(bpy.data.materials['Fabric035'])
  for f in obj.data.polygons:
    f.use_smooth = True
  smocked_collection = bpy.data.collections.new('cloth_sim_collection')
  bpy.context.scene.collection.children.link(smocked_collection)
  smocked_collection.objects.link(obj)

  # Add stitching edges.
  # Correspondence to vertices in the coarse grid.
  vid = [np.argmin(np.linalg.norm(V - p, axis=1)) for p in fsp.V]
  # Stitching lines constraints in the fine grid.
  stitching_edges = get_stitching_lines_in_new_grid(fsp, vid)
  bpy.context.view_layer.objects.active = obj
  bpy.ops.object.mode_set(mode='EDIT')
  bm = bmesh.from_edit_mesh(obj.data)
  bm.verts.ensure_lookup_table()
  for e in stitching_edges:
    bm.edges.new((bm.verts[e[0]], bm.verts[e[1]]))

  bpy.ops.object.mode_set(mode='OBJECT')

  add_text_to_scene(body="Blender cloth simulation", 
                          location=np.array(obj.location) + (LAYOUT_TEXT*2.7, np.max(fsp.V[:, 1]) + 1, 0), 
                          scale=(1,1,1),
                          obj_name="cloth_sim_annotation",
                          coll_name="cloth_sim_collection")

  return obj
  
def add_cloth_sim(obj):
  bpy.context.view_layer.objects.active = obj
  bpy.ops.object.modifier_add(type='CLOTH')
  obj.modifiers['Cloth'].settings.tension_damping = 5.0
  obj.modifiers['Cloth'].settings.tension_stiffness = 2.6
  obj.modifiers['Cloth'].settings.tension_stiffness_max = 27.19999885559082
  obj.modifiers['Cloth'].settings.use_sewing_springs = True
  obj.modifiers['Cloth'].settings.compression_stiffness = 1.0
  obj.modifiers['Cloth'].settings.bending_stiffness = 8
  obj.modifiers['Cloth'].settings.use_pressure = True
  obj.modifiers['Cloth'].settings.gravity = Vector((0, 0, 0))
  obj.modifiers['Cloth'].collision_settings.use_collision = True
  obj.modifiers['Cloth'].collision_settings.use_self_collision = True
  obj.modifiers["Cloth"].settings.uniform_pressure_force = 0.01
  obj.modifiers["Cloth"].settings.sewing_force_max = 0.8
  obj.modifiers["Cloth"].settings.effector_weights.gravity = 0
  obj.modifiers["Cloth"].settings.shrink_min = 0.06




