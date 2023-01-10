bl_info = {
    "name": "SmockingDesign",
    "author": "Jing Ren",
    "version": (1, 0),
    "blender": (2, 80, 0),
    "location": "View3D > N",
    "description": "Design and Simulate Smocking Arts",
    "warning": "",
    "doc_url": "",
    "category": "Design",
}



import bpy
import bmesh
from bpy.types import Operator
from bpy.types import (Panel, Operator)
import sys
import os

# TODO: Find a better way to load the module...
import importlib
FILE_PATH = os.path.dirname(bpy.context.space_data.text.filepath)
sys.path.append(FILE_PATH)
import cpp_smocking_solver
import arap
import arap_stitching_lines
import cloth_sim
import honeycomb
import wrapping
importlib.reload(cpp_smocking_solver)
importlib.reload(arap)
importlib.reload(arap_stitching_lines)
importlib.reload(cloth_sim)
importlib.reload(honeycomb)
importlib.reload(wrapping)

import math
from mathutils import Vector
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
from scipy.optimize import Bounds

import itertools

import os
from datetime import datetime
import time


# install scipy to blender's python
# ref: https://blender.stackexchange.com/questions/5287/using-3rd-party-python-modules
# cd /path/to/blender/python/bin
# ./python -m ensurepip
# ./python -m pip install scipy

# ========================================================================
#                          Global Variables
# ========================================================================

col_blue = (76/255.0, 201/255.0,240/255.0)
col_yellow = (254/255.0, 228/255.0, 64/255.0)
col_green = (181/255.0, 228/255.0, 140/255.0)
col_red = (240/255.0, 113/255.0, 103/255.0)
col_gray = (0.75, 0.75, 0.75)

STROKE_SIZE = 10 # to plot the stiching lines

LAYOUT_TEXT = 0.4 # the space above the pattern
# align P1/P2/P1_P2_COMBINED
LAYOUT_Y_SHIFT = 2
LAYOUT_X_SHIFT = 2
# align USP FSP
LAYOUT_USP_FSP_SPACE = 2

# collection names
COLL_NAME_USP = "UnitSmockingPattern"
COLL_NAME_USP_SL = "UnitStitchingLines"
COLL_NAME_FSP = "SmockingPattern"
COLL_NAME_FSP_SL = "StitchingLines"
COLL_NAME_P1 = "FSP1"
COLL_NAME_P1_SL = "FSP1_SL"
COLL_NAME_P2 = "FSP2"
COLL_NAME_P2_SL = "FSP2_SL"

COLL_NAME_FSP_TMP = "FSP_tmp"
COLL_NAME_FSP_TMP_SL = "FSP_tmp_SL"


COLL_NAME_SG = "SmockedGraph"
COLL_NAME_SG_SL = "SmockedGraphStrokes"
MESH_NAME_SG = "Graph"

MESH_NAME_USP = "Grid"
MESH_NAME_FSP = "FullPattern"
MESH_NAME_P1 = 'Pattern01'
MESH_NAME_P2 = 'Pattern02'



# global variables from user input

PROPS = [
    ('if_highlight_button', bpy.props.BoolProperty(name="highlight buttons", default=True)),
    # for unit grid
    ('base_x', bpy.props.IntProperty(name='X', default=3, min=1, max=20)),
    ('base_y', bpy.props.IntProperty(name='Y', default=3, min=1, max=20)),
    # import/export stitching lines
    ('path_export', bpy.props.StringProperty(subtype='DIR_PATH', name='Path',default='/tmp/')),
    ('filename_export', bpy.props.StringProperty(name='Name', default='my_pattern_name')),
    ('path_import', bpy.props.StringProperty(subtype='FILE_PATH', name='File')),
    # for full grid (full smocking pattern)
    ('num_x', bpy.props.IntProperty(name='Repeat along X-axis', default=3, min=1, max=20)),
    ('num_y', bpy.props.IntProperty(name='Repeat along Y-axis', default=3, min=1, max=20)),
    ('shift_x', bpy.props.IntProperty(name='Shift along X-axis', default=0, min=-10, max=10)),
    ('shift_y', bpy.props.IntProperty(name='Shift along Y-axis', default=0, min=-10, max=10)),
#    ('type_tile', bpy.props.EnumProperty(items = [("regular", "regular", "tile in a regular grid"), ("radial", "radial", "tile in a radial grid")], name="Type", default="regular")),
    ('margin_top', bpy.props.FloatProperty(name='Top', default=0, min=0, max=10)),
    ('margin_bottom', bpy.props.FloatProperty(name='Bottom', default=0, min=0, max=10)),
    ('margin_left', bpy.props.FloatProperty(name='Left', default=0, min=0, max=10)),
    ('margin_right', bpy.props.FloatProperty(name='Right', default=0, min=0, max=10)),
    # export the full smocking pattern as obj
    ('path_export_fullpattern', bpy.props.StringProperty(subtype='FILE_PATH', name='Path', default='/tmp/')),
    ('path_import_fullpattern', bpy.props.StringProperty(subtype='FILE_PATH', name='Path', default='/tmp/my_pattern_name.obj')),
    ('filename_export_fullpattern', bpy.props.StringProperty(name='Name', default='my_pattern_name')),
    ('export_format', bpy.props.EnumProperty(items = [(".obj", "OBJ", ".obj"), (".off", "OFF", ".off")], name="Format", default=".obj")),
    # FSP: combine two patterns
    ('file_import_p1', bpy.props.StringProperty(subtype='FILE_PATH', name='P1 path')),
    ('file_import_p2', bpy.props.StringProperty(subtype='FILE_PATH', name='P2 path')),
    ('combine_direction', bpy.props.EnumProperty(items = [("x", "horizontally", "along x axis", 'EVENT_H',0), 
                                                          ("y", "vertically", "along y axis",'EVENT_V',1)], 
                                                          name="Combine", default="x")),
    ('combine_space', bpy.props.IntProperty(name='Spacing', default=2, min=1, max=20)),
    ('combine_shift', bpy.props.IntProperty(name='Shift', default=0, min=-20, max=20)),
    ('fsp_edit_selection', bpy.props.EnumProperty(items= (('V', 'VERT', 'move vertices', 'VERTEXSEL', 0),    
                                                          ('E', 'EDGE', 'move edges', 'EDGESEL', 1),    
                                                          ('F', 'FACE', 'add/delete faces', 'FACESEL', 2)) ,  
                                   default="F",
                                   name = "Select",  
                                   description = "")),
    ('radial_grid_ratio', bpy.props.FloatProperty(name='ratio', default=0.9, min=0.1, max=1)), 
    # Embedding
    ('pleat_eq', bpy.props.FloatProperty(name='Pleats equality constraint weight', default=1e3, min=0.0)), 
    ('pleat_max_embed', bpy.props.FloatProperty(name='Max embedding weight', default=1, min=0.0)), 
    ('pleat_variance', bpy.props.FloatProperty(name='Pleat height variance weight', default=1, min=0.0)), 
    ('arap_constraint_weight', bpy.props.FloatProperty(name='Constraints weight', default=1, min=0.0)), 
    ('arap_num_iters', bpy.props.IntProperty(name='Num iters', default=100, min=0)), 
    # Smocked graph

    ('graph_select_type', bpy.props.EnumProperty(items= (('V', 'VERT', 'highlight vertices', 'VERTEXSEL', 0),    
                                                          ('E', 'EDGE', 'highlight edges', 'EDGESEL', 1)),    
                                   default="V",
                                   name = "Select",  
                                   description = "")),
    ('graph_highlight_type', bpy.props.EnumProperty(items= (('all', 'all', 'show all vtx/edge'),
                                                      ('underlay', 'underlay', 'highlight underlay'),    
                                                      ('pleat', 'pleat', 'hightlight pleat')) ,  
                                   default="all",
                                   name = "Highlight",  
                                   description = "")),
    ('opti_init_type', bpy.props.EnumProperty(items= (('center', 'center', 'center of stitching lines'),    
                                                      ('random', 'random', 'random initialization')) ,  
                                   default="center",
                                   name = "Initialization",  
                                   description = "Initialization")),
    ('opti_init_pleat_height', bpy.props.FloatProperty(name="Initial Pleat Height", default=1, min=0.1, max=5)),
    
    ('opti_dist_preserve', bpy.props.EnumProperty(items= (('exact', 'exact', 'exact'),    
                                                      ('approx', 'approximation', 'approximation')) ,  
                                   default="exact",
                                   name = "Constraints",  
                                   description = "Constraints")),

    ('opti_if_add_delaunay', bpy.props.BoolProperty(name="Add Constraints from Delaunay", default=False)),

    ('opti_mtd', bpy.props.EnumProperty(items= (('Newton', 'Newton-CG', 'Newton Conjugated Gradient'),    
                                                ('BFGS', 'BFGS', 'BFGS')) ,  
                                   default="Newton",
                                   name = "Solver",  
                                   description = "Solver")),

    ('opti_tol', bpy.props.FloatProperty(name="Tolerance", default=1e-3, min=1e-12, max=0.1)),
    ('opti_maxIter', bpy.props.IntProperty(name="MaxIter", default=100, min=1, max=100000)),
    ('opti_if_display', bpy.props.BoolProperty(name="Print convergence messages", default=True))

    
    ]
    
    
# to extract the stitching lines from user selection

class GlobalVariable():
    currentDrawing = []
    # so we dont show multiple current drawings when repeated click "Done"
    if_curr_drawing_is_shown = False 
    if_user_is_drawing = True
    savedStitchingLines = []
    colSaved = col_blue
    colTmp = col_yellow
    runtime_log = []



# data for optimization
class SolverData():
    unit_smocking_pattern = []
    full_smocking_pattern = []
    smocked_graph = None
    embeded_graph = None
    arap_data = None
    smocking_design = []

    # temporary data
    tmp_fsp1 = []
    tmp_fsp2 = []
    tmp_fsp = []



class OptiData():
    C_underlay_eq = []
    C_underlay_neq = []
    C_pleat_eq = []
    C_pleat_neq = []
    weights = {'w_underlay_eq': 1e5, 
               'w_underlay_neq': 1e2,
               'eps_enq': -1e-6,
               'w_pleat_eq': 1e5,
               'w_pleat_neq': 1e3}


# ========================================================================
#                         classes for the solver
# ========================================================================


class debug_clear(Operator):
    bl_idname = "object.debug_clear"
    bl_label = "clear data in scene"
    
    def execute(self, context):
        initialize_collections()
        initialize_data()

        return {'FINISHED'}

class debug_render(Operator):
  bl_idname = "object.debug_render"
  bl_label = "Render the smocking design"
  
  def execute(self, context):
    dt = bpy.types.Scene.solver_data
    g_V = dt.arap_data[0]
    max_z = g_V[2,:].max()
    bbox_min, bbox_max = (np.amin(g_V, axis=1), np.amax(g_V, axis=1))
    center = (bbox_max + bbox_min) / 2
    camLocation = center
    camLocation[2] = max_z + max((bbox_max - bbox_min)/2) + 7
    if 'camera' in bpy.data.objects:
      camera = bpy.data.objects['camera']
      camera.location = camLocation
    else:
      bpy.ops.object.camera_add(location = camLocation)
      camera = bpy.context.object
      camera.name = 'camera'
    camera.data.lens = 20
    # Look at.
    direction = Vector(center) - Vector(camLocation)
    rotQuat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rotQuat.to_euler()
    bpy.ops.render.render()
    return {'FINISHED'}

class isometric_distortion(Operator):
  bl_idname = "object.isometric_distortion"
  bl_label = "The isometric distortion."
  
  def execute(self, context):
    dt = bpy.types.Scene.solver_data
    # If there's an active cloth simulation.
    if "cloth_sim_obj" in bpy.data.objects:
      obj = bpy.data.objects["cloth_sim_obj"]
      bpy.context.view_layer.objects.active = obj
      grid_verts = [list(v.co) for v in obj.data.vertices]
      bpy.ops.object.modifier_apply(modifier="Cloth")
      deform_verts = [list(v.co) for v in obj.data.vertices]
      faces = [list(p.vertices) for p in obj.data.polygons]
      distortion = cpp_smocking_solver.get_isometry_distortion(grid_verts, deform_verts, faces)
      print(distortion)

    # If there's an active arap.
    if dt.arap_data is not None and dt.arap_data[2].name in bpy.data.meshes:
      mesh = dt.arap_data[2]
      grid_verts = dt.arap_data[4].get_points().T
      deform_verts = [list(v.co) for v in mesh.vertices]
      faces = [list(p.vertices) for p in mesh.polygons]
      distortion = cpp_smocking_solver.get_isometry_distortion(grid_verts, deform_verts, faces)
      print(distortion)

    return {'FINISHED'}

class debug_print(Operator):
    bl_idname = "object.debug_print"
    bl_label = "print data in scene"
    
    def execute(self, context):
        
        props = bpy.types.Scene.sl_props
        dt = bpy.types.Scene.solver_data
        
        usp = dt.unit_smocking_pattern
        fsp = dt.full_smocking_pattern
        print('--------------------------------------------------------')
        print('- data stored in current scene:')
        print('--------------------------------------------------------')
        if usp and fsp:
            usp.info()
            fsp.info()
            
        else:
            print('Import the smokcing pattern first')
            
                
        return {'FINISHED'}

def opti_energy_maximize_embedding(X):
    D = squareform(pdist(X,'euclidean'))  
    fval = -np.sum(np.sum(D))
    return fval


def opti_energy_pleat_height_varicance(X_pleat):
    fval = np.var(X_pleat[:,2])
    return fval


def opti_energy_preserve_edge_length(X, C_eq):
    I = C_eq[:,0].astype(int)
    J = C_eq[:,1].astype(int)
    d_eq = np.sqrt(np.sum(np.power(X[I, :] - X[J, :], 2),axis = 1))
    fval = sum(np.power(d_eq - C_eq[:,2], 2))
    return fval



def opti_energy_sg_underlay(x_in, C_eq):
    # energy to embedding the underaly graph of the smocked graph
    X = x_in.reshape(int(len(x_in)/2), 2) # x_in the flattened xy-coordinates of the underaly graph
    return opti_energy_preserve_edge_length(X, C_eq)


def opti_energy_sg_pleat(x_pleat_in, 
                         X_underlay, 
                         C_pleat_eq,
                         w_embed=1,
                         w_eq=1e4,
                         w_var = 1):
    if len(X_underlay[0]) == 2:    
        X_underlay = np.concatenate((X_underlay, np.zeros((len(X_underlay),1))), axis=1)

    X_pleat = x_pleat_in.reshape(int(len(x_pleat_in)/3), 3)

    X = np.concatenate((X_underlay, X_pleat), axis = 0)

    fval = w_embed*opti_energy_maximize_embedding(X) +\
           w_eq*opti_energy_preserve_edge_length(X, C_pleat_eq) +\
           w_var*opti_energy_pleat_height_varicance(X_pleat)

    return fval


class debug_func(Operator):
    bl_idname = "object.debug_func"
    bl_label = "function to test"
    
    def execute(self, context):
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Debugging....Current Time =", current_time)


        
        initialize_collections()
        initialize_data()

    
        context.scene.path_import = '/Users/avivsegall/projs/SmockingDesign/unit_smocking_patterns/braid.usp'
        bpy.ops.object.import_unit_pattern()

        bpy.context.scene.num_x = 5
        bpy.context.scene.num_y = 5
                
        bpy.ops.object.create_full_smocking_pattern()

        
        bpy.ops.object.sg_draw_graph()


        dt = bpy.types.Scene.solver_data


        fsp = dt.full_smocking_pattern
        sg = dt.smocked_graph
    
        C_underlay_eq, C_pleat_eq = sg.return_distance_constraint()


        #----------------- embed the underlay 
        X_underlay_ini = sg.V[sg.vid_underlay, 0:2]    

    
        x0 = X_underlay_ini.flatten()
        
        start_time = time.time()
        res_underlay = np.array(cpp_smocking_solver.embed_underlay(X_underlay_ini, C_underlay_eq))
        '''
        res_underlay = minimize(opti_energy_sg_underlay, 
                           x0, 
                           method='Nelder-Mead',
                           args=(C_underlay_eq), 
                           options=opti_get_NelderMead_solver_options())
        '''
        msg = 'optimization: embed the underlay graph: %f second' % (time.time() - start_time)
        bpy.types.Scene.sl_props.runtime_log.append(msg)

        X = res_underlay.reshape(int(len(res_underlay)),2)

        X_underlay = np.concatenate((X, np.zeros((len(X),1))), axis=1)

        #------------------ embed the pleat
        X_pleat = sg.V[sg.vid_pleat, 0:2]

        # option 01: initialize from the original position 
        X_pleat = np.concatenate((X_pleat, np.ones((len(X_pleat),1))), axis=1)
        
        x0 = X_pleat.flatten()
        w_pleat_embed = 1
        w_pleat_eq = 1e3
        w_pleat_var = 1
        start_time = time.time()
        res_pleat = np.array(cpp_smocking_solver.embed_pleats(
          X_underlay, X_pleat, C_pleat_eq, w_pleat_var, w_pleat_embed, w_pleat_eq))
        '''
        res_pleat = minimize(opti_energy_sg_pleat, 
                           x0, 
                           method='Nelder-Mead',
                           args=(X_underlay, C_pleat_eq, w_pleat_embed, w_pleat_eq, w_pleat_var), 
                           options=opti_get_NelderMead_solver_options())
        '''
        msg = 'optimization: embed the underlay graph: %f second' % (time.time() - start_time)
        bpy.types.Scene.sl_props.runtime_log.append(msg)

        X_pleat = res_pleat.reshape(int(len(res_pleat)), 3)



        X_all = np.concatenate((X_underlay, X_pleat), axis=0)

        print_runtime()

        # Run arap.
        g_arap, g_constraints, g_F, UV = prepare_arap(X_all, fsp, sg)
        g_V = g_arap(g_constraints, 0)
        g_iters = 100

        # Show the result.
        smocked_mesh = bpy.data.meshes.new("Smocked")
        smocked_mesh.from_pydata(g_V.T,[],g_F)
        
        smocked_mesh.update()
        g_mesh = smocked_mesh
        dt.arap_data = [g_V, g_F, g_mesh, g_iters, g_arap, g_constraints]

        # Add uv layer
        uvlayer = smocked_mesh.uv_layers.new()
        for face in smocked_mesh.polygons:
          for vert_idx, loop_idx in zip(face.vertices, face.loop_indices):
              uvlayer.data[loop_idx].uv = (UV[vert_idx, 0] / 20.0, UV[vert_idx, 1] / 20.0)
        
        obj = bpy.data.objects.new('Smocked mesh', smocked_mesh)
        obj.location = (0, 0, 4)
        obj.data.materials.append(bpy.data.materials['Fabric035'])
        for f in obj.data.polygons:
          f.use_smooth = True
        smocked_collection = bpy.data.collections.new('smocked_collection')
        bpy.context.scene.collection.children.link(smocked_collection)
        smocked_collection.objects.link(obj)

        # add_text_to_scene(body="Simulated Smocked Mesh", 
        #                   location=(0, -32, 0), 
        #                   scale=(1,1,1),
        #                   obj_name='grid_annotation',
        #                   coll_name=COLL_NAME_USP)


        # trans = [0,-30, 0]
        # for eid in range(sg.ne):
        #     vtxID = sg.E[eid, :]
        #     pos = X_all[vtxID, :] + trans
        #     if sg.is_edge_underlay(eid):
        #         col = col_gray
        #     else:
        #         col = col_red
        #     draw_stitching_line(pos, col, "embed_" + str(eid), int(STROKE_SIZE/2), COLL_NAME_SG)
        


        # X = X_underlay
        # trans = [0,-20, 0]
        # for eid in sg.eid_underlay:
        #     vtxID = sg.E[eid, :]
        #     pos = X[vtxID, :] + trans
        #     draw_stitching_line(pos, col_red, "embed_underlay2_" + str(eid), int(STROKE_SIZE/2), COLL_NAME_SG)


        '''
        #----------------- plot the embeded graph
        
        
        

        for vid in range(len(X)):
            pos = X[vid, :] + trans
            add_text_to_scene(body='v'+str(vid), 
                              location=tuple(pos), 
                              scale=(1,1,1),
                              obj_name='v'+str(vid),
                              coll_name=COLL_NAME_SG)

        
        # X = X_underlay
        trans = [0,-30, 0]
        for eid in range(sg.ne):
            vtxID = sg.E[eid, :]
            pos = X_all[vtxID, :] + trans
            if sg.is_edge_underlay(eid):
                col = col_gray
            else:
                col = col_red
            draw_stitching_line(pos, col, "embed_" + str(eid), int(STROKE_SIZE/2), COLL_NAME_SG)
        
        '''
        return {'FINISHED'}


def opti_energy_sg_pleat_old(x_pleat_in,
                         X_underlay,
                         C_pleat_neq, 
                         w_pleat_neq=1e6, 
                         w_var = 1e1,
                         eps_neq=-1e-3,
                         if_return_all_terms=False):

    if len(X_underlay[0]) == 2:    
        X_underlay = np.concatenate((X_underlay, np.zeros((len(X_underlay),1))), axis=1)

    X_pleat = x_pleat_in.reshape(int(len(x_pleat_in)/3), 3)

    X = np.concatenate((X_underlay, X_pleat), axis = 0)
    
    D1 = squareform(pdist(X,'euclidean'))  
    
    # maximize the embedding: such that the vertices are far from eath other
    fval_max = -np.sum(np.sum(D1))

    # make usre the inequality is satisfied
    d_neq = get_mat_entry(D1, C_pleat_neq[:,0], C_pleat_neq[:, 1])
    # fval_neq = sum(d_neq - C_pleat_neq[:,2] > eps_neq)
    fval_neq = sum(np.power(d_neq - C_pleat_neq[:,2], 2))


    # penalize the variance of the height
    fval_var = np.var(X_pleat[:,2])

    fval = fval_max + w_pleat_neq*fval_neq + w_var*fval_var

    # print(fval)
    if if_return_all_terms: # for debug
        return fval_max, fval_neq, fval_var
    else:
        return fval



# TODO: need to update according the user input
def opti_get_BFGS_solver_options():
    bfgs_options =  {'disp': True, 
                   'verbose':1, 
                   'xtol':1e-6, 
                   'ftol':1e-6, 
                   'maxiter':1e6, 
                   'maxfun':1e6}

    return bfgs_options


# TODO: need to update according the user input
def opti_get_NelderMead_solver_options():
    
    nm_options =  {'disp': True, 
                   'verbose':1, 
                   'xtol':1e-6, 
                   'ftol':1e-6, 
                   'maxiter':1e6, 
                   'maxfun':1e6}

    return nm_options

def opti_energy_sg_underlay_old(x_in,
                            C_eq, 
                            C_neq, 
                            w_eq=1e2,
                            w_neq=1e6, 
                            eps_neq=-1e-3,
                            if_return_all_terms=False):
    # energy to embedding the underaly graph of the smocked graph
    x = x_in.reshape(int(len(x_in)/2), 2) # x_in the flattened xy-coordinates of the underaly graph
    D1 = squareform(pdist(x,'euclidean'))

    # maximize the embedding: such that the vertices are far from eath other
    fval_max = -np.sum(np.sum(D1))

    # make sure the equality is satisfied
    d_eq = get_mat_entry(D1, C_eq[:, 0], C_eq[:, 1])
    fval_eq = sum(np.power(d_eq - C_eq[:,2], 2))

    # make usre the inequality is satisfied
    d_neq = get_mat_entry(D1, C_neq[:,0], C_neq[:, 1])
    fval_neq = sum(d_neq - C_neq[:,2] > eps_neq)

    fval = fval_max + w_eq*fval_eq + w_neq*fval_neq
    # print(fval)
    if if_return_all_terms: # for debug
        return fval_max, fval_eq, fval_neq
    else:
        return fval



# TOOD: 
def SG_find_valid_underlay_constraints_approx(sg, D):
    # find inexact distance constraints if the underlay graph is too large
    # check every triplet of vertices can be time consuming

    print('Not done yet:/')




def SG_find_valid_underlay_constraints_exact(sg, D):
    # for the smocked graph (sg)
    # find the valid the distance constraints in exact way
    # i.e., consider all pairs of vertices

    # input: D stores the pairwise distance constraint for the underlay graph
    # extracted from the smocking pattern (to make sure the fabric won't break)
    # but not all of them are useful for embedding optimization

    start_time = time.time()
        

    # all combinations of (i,j,k) - a triplet of three vertices
    c = nchoosek(range(sg.nv_underlay), 3)
    # we then check how many of them are useless
    # in exact way
    useless_constr = []
    for i, j, k in zip(c[:,0], c[:,1], c[:,2]):
        if D[i,j] + D[i, k] < D[k, j]:
            useless_constr.append([k,j])

        if D[i,j] + D[k,j] <  D[i,k]:
            useless_constr.append([i,k])

        if D[i,k] + D[k,j] < D[i,j]:
            useless_constr.append([i,j])


    # remove redundant pairs
    useless_constr = sort_edge(useless_constr)

    # all vertex pairs
    A = nchoosek(range(sg.nv_underlay), 2)
    
    idx_diff, _ = setdiffnd(A, useless_constr)
    
    constrained_vtx_pair = A[idx_diff]
    msg = 'runtime: find constraints (exact): %f second' % (time.time() - start_time)
    bpy.types.Scene.sl_props.runtime_log.append(msg)

    return constrained_vtx_pair


    

# ========================================================================
#                         classes for the solver
# ========================================================================


#----------------------- Smocked Graph Class --------------------------------

class SmockedGraph():
    """the smocked graph from the full smocking pattern (fsp)"""

    def __init__(self, fsp, init_type='center'):
        """extract the smocked graph from the fsp"""
        
        self.V, \
        self.E, \
        self.F, \
        self.dict_sg2sp, \
        self.dict_sp2sg, \
        self.vid_underlay, \
        self.vid_pleat, \
        self.eid_underlay, \
        self.eid_pleat = extract_smocked_graph_from_full_smocking_pattern(fsp, init_type ='center')

        self.nv = len(self.V)
        self.ne = len(self.E)
        self.nv_pleat = len(self.vid_pleat)
        self.nv_underlay = len(self.vid_underlay)
        self.full_smocking_pattern = fsp



    def return_max_dist_constraint_for_vtx_pair(self, vid1, vid2):
        """Compute the maximum distance between the two nodes"""

        fsp = self.full_smocking_pattern
        vid1_sp = self.dict_sg2sp[vid1]
        vid2_sp = self.dict_sg2sp[vid2]
        d = cdist(fsp.V[vid1_sp, :], fsp.V[vid2_sp, :])
        return np.min(d)


    def return_max_dist_constraint_for_edge(self, eid):
        return self.return_max_dist_constraint_for_vtx_pair(self.E[eid, 0], self.E[eid, 1])




    def return_distance_constraint(self):
        # prepare the equality constraints for the underlay    
        C_underlay_eq = []
        for eid in self.eid_underlay:
            d = self.return_max_dist_constraint_for_edge(eid)
            C_underlay_eq.append([self.E[eid, 0], self.E[eid, 1], d])

        # prepare the equality constraints for the pleat
        C_pleat_eq = []
        for eid in self.eid_pleat:
            d = self.return_max_dist_constraint_for_edge(eid)
            C_pleat_eq.append([self.E[eid, 0], self.E[eid, 1], d])
        return np.array(C_underlay_eq), np.array(C_pleat_eq)



    def return_pairwise_distance_constraint_for_underlay(self):
        num = self.nv_underlay
        D = np.zeros((num, num))
        for i in range(num-1):
            for j in range(1,num):
                dist = self.return_max_dist_constraint_for_vtx_pair(i,j)
                D[i,j] = dist
                D[j,i] = dist
        return D


    def find_vtx_neighboring_edges(self, vid):
        eids = find_index_in_list(self.E[:, 0].tolist(), vid)
        eids.append(find_index_in_list(self.E[:, 1].tolist(), vid))
        
        return np.unique(eids)


    def is_vtx_pleat(self, vid):
        return vid in self.vid_pleat



    def is_vtx_underlay(self, vid):
        return vid in self.vid_underlay



    def is_edge_pleat(self, eid):
        return eid in self.eid_pleat

    
    def is_edge_underlay(self, eid):
        return eid in self.eid_underlay



    def info(self):
        print('\n----------------------- Info: Smocked Graph -----------------------')
        print('No. vertices %d : %d underlay + %d pleat' % (self.nv, len(self.vid_underlay), len(self.vid_pleat) ) )
        print('No. edges %d : %d underlay + %d pleat' % (self.ne, len(self.eid_underlay), len(self.eid_pleat) ) )
        print('-------------------------------------------------------------------\n')


        
        
    def plot(self, 
             location=(0,0), 
             if_show_separate_underlay=True,
             if_show_separate_pleat=True,
             if_debug=True):
        initialize_pattern_collection(COLL_NAME_SG, COLL_NAME_SG_SL)        
        construct_object_from_mesh_to_scene(self.V, self.F, MESH_NAME_SG, COLL_NAME_SG)
        mesh = bpy.data.objects[MESH_NAME_SG]
        
        mesh.scale = (1, 1, 1)
        mesh.location = (location[0]-min(self.V[:,0]), location[1]-max(self.V[:,1])+min(self.V[:,1])-LAYOUT_Y_SHIFT, 0)
        mesh.show_axis = False
        mesh.show_wire = True
        mesh.display_type = 'WIRE'
        select_one_object(mesh)


        text_loc = (location[0], location[1]-LAYOUT_Y_SHIFT+LAYOUT_TEXT, 0)
        add_text_to_scene(body="Smocked Graph", 
                          location=text_loc, 
                          scale=(1,1,1),
                          obj_name='graph_annotation',
                          coll_name=COLL_NAME_SG)

        # visualize the vertices in different colors
        fsp = self.full_smocking_pattern
        # show underlay 
        if if_show_separate_underlay:

            trans = np.array((fsp.return_pattern_width()+2,0,0))

            for eid in self.eid_underlay:
                vtxID = self.E[eid, :]
                pos = get_vtx_pos(mesh, np.array(vtxID)) + trans
                draw_stitching_line(pos, col_red, "edge_underlay_" + str(eid), STROKE_SIZE, COLL_NAME_SG)
    #       
            add_text_to_scene(body="Underaly Graph", 
                              location=tuple(np.array(text_loc) + trans), 
                              scale=(1,1,1),
                              obj_name='underlay_graph_annotation',
                              coll_name=COLL_NAME_SG)

            if if_debug: # add vtxID
                for vid in self.vid_underlay:
                    pos = get_vtx_pos(mesh, [vid]) + trans
                    add_text_to_scene(body='v'+str(vid), 
                                      location=tuple(pos[0]), 
                                      scale=(1,1,1),
                                      obj_name='v'+str(vid),
                                      coll_name=COLL_NAME_SG)



        
        if if_show_separate_pleat:
            
            trans = np.array((fsp.return_pattern_width()+2, -fsp.return_pattern_height()-2,0))

            # first add underlay
            for eid in self.eid_underlay:
                vtxID = self.E[eid, :]
                pos = get_vtx_pos(mesh, np.array(vtxID)) + trans
                draw_stitching_line(pos, col_gray, "edge_underlay2_" + str(eid), int(STROKE_SIZE/2), COLL_NAME_SG)


            

            for eid in self.eid_pleat:
                vtxID = self.E[eid, :]
                pos = get_vtx_pos(mesh, np.array(vtxID)) + trans
                draw_stitching_line(pos, col_red, "edge_pleat_" + str(eid), STROKE_SIZE, COLL_NAME_SG)
    #       
            add_text_to_scene(body="Pleat Graph", 
                              location=tuple(np.array(text_loc) + trans), 
                              scale=(1,1,1),
                              obj_name='pleat_graph_annotation',
                              coll_name=COLL_NAME_SG)
        





#----------------------- Unit Smocking Pattern Class ----------------------------



class UnitSmockingPattern():
    """create a unit pattern"""
        
    def __init__(self, 
                base_x, 
                base_y,
                stitching_lines,
                stitching_points,
                stitching_points_line_id):
        self.base_x = base_x
        self.base_y = base_y
        self.stitching_lines = stitching_lines
        self.stitching_points = stitching_points
        self.stitching_points_line_id = stitching_points_line_id
        
        
    def get_pts_in_stitching_line(self, lid):
        pids = find_index_in_list(self.stitching_points_line_id, lid)
        pos = np.array(self.stitching_points)[pids, :]
        
        return pos

    def get_vtx_in_stitching_line(self, lid):
        return np.array(self.stitching_lines[lid])
    
    def plot(self, location=(0,0)):
        
        initialize_pattern_collection(COLL_NAME_USP, COLL_NAME_USP_SL)

        generate_grid_for_unit_pattern(self.base_x, self.base_y)
    
        mesh = bpy.data.objects[MESH_NAME_USP]
    
        mesh.location = (location[0], location[1], 0)

        add_text_to_scene(body="Unit Smocking Pattern", 
                          location=(0, location[1] + self.base_y + LAYOUT_TEXT, 0), 
                          scale=(1,1,1),
                          obj_name='grid_annotation',
                          coll_name=COLL_NAME_USP)
        
        for lid in range(len(self.stitching_lines)):

            vtxID = self.get_vtx_in_stitching_line(lid)

            pos = get_vtx_pos(mesh, np.array(vtxID))
            print(pos)
 
            draw_stitching_line(pos, col_blue, "stitching_line_" + str(lid), STROKE_SIZE, COLL_NAME_USP_SL)


    def info(self):
        print('\n------------------- Info: Unit Smocking Pattern -------------------')
        print('base_x: ' + str(self.base_x) + ', base_y: ' + str(self.base_y))
        print('No. stitching lines: ' + str(len(self.stitching_lines)))
        print('-------------------------------------------------------------------\n')



#----------------------- Smocking Pattern Class --------------------------------

class SmockingPattern():
    """Full Smocking Pattern"""
    def __init__(self, V, F, E,
                 stitching_points,
                 stitching_points_line_id,
                 stitching_points_patch_id=[],
                 stitching_points_vtx_id=[],
                 pattern_name = "FllPattern", 
                 coll_name=COLL_NAME_FSP,
                 stroke_coll_name = COLL_NAME_FSP_SL,
                 annotation_text="Full Smocking Pattern"):
        # the mesh for the smocking pattern
        self.V = V # can be 2D or 3D vtx positions
        self.F = F
        self.E = E
        self.nv = len(V)
        self.nf = len(F)
        self.ne = len(E)
        
        # the stitching points: in 2D/3D positions
        self.stitching_points = np.array(stitching_points)
        if self.stitching_points.shape[0] > 0:
          self.stitching_points = self.stitching_points[:,0:len(self.V[0])]

        # the lineID of the stitching points
        # the points with the same line ID will be sew together
        self.stitching_points_line_id = np.array(stitching_points_line_id)
        
        # the patchID of the stitching points from the tiling process
        # save this information for visualization only
        # not useful for optimization
        if len(stitching_points_patch_id) > 0:
            self.stitching_points_patch_id = np.array(stitching_points_patch_id)
        else: # do not have it, we then use the line_id
            self.stitching_points_patch_id = np.array(stitching_points_line_id)
        
        # the vtxID of each stitching points in V
        if len(stitching_points_vtx_id) > 0:
            self.stitching_points_vtx_id = np.array(stitching_points_vtx_id)
        else:
            self.get_stitching_points_vtx_id()
        
        self.pattern_name = pattern_name
        self.annotation_text = annotation_text
        self.coll_name = coll_name
        self.stroke_coll_name = stroke_coll_name



    def get_stitching_points_vtx_id(self):
        all_sp_vid = []
        for ii in range(len(self.stitching_points)):
          # vid = find_matching_rowID(self.V, self.stitching_points[ii,0:len(self.V[0])])
          vid = find_matching_rowID(self.V, self.stitching_points[ii,:])
          if len(vid[0]) == 0:
            continue
#            print(self.stitching_points[ii, 0:2])
            
          all_sp_vid.append(vid[0][0].tolist())
        self.stitching_points_vtx_id = np.array(all_sp_vid)


    
    def update_mesh(self, V_new, F_new, E_new):
        self.V = V_new
        self.F = F_new
        self.E = E_new
        # update the stitching points ID since the V is updated
        self.get_stitching_points_vtx_id()



    
    def update_stitching_lines(self, all_sp, all_sp_lid, all_sp_pid):
        self.stitching_points = np.array(all_sp)
        self.stitching_points_line_id = np.array(all_sp_lid)
        self.stitching_points_patch_id = np.array(all_sp_pid)
        self.get_stitching_points_vtx_id()
     

    def move_to_origin(self):
        trans = [min(self.V[:, 0]), min(self.V[:, 1]), 0]
        self.V = self.V - trans[0:len(self.V[0])]
        self.stitching_points = self.stitching_points - trans[0:len(self.stitching_points[0])]


    def get_vid_in_stitching_line(self, lid):
        pids = find_index_in_list(self.stitching_points_line_id, lid)
        vtxID = self.stitching_points_vtx_id[pids]
        
        return vtxID
    


    def get_pts_in_stitching_line(self, lid):
    
        pids = find_index_in_list(self.stitching_points_line_id, lid)
        pos = np.array(self.stitching_points)[pids, :]
        
        return pos, pids    
    
    def num_stitching_lines(self):
         
        return int(max(self.stitching_points_line_id)) + 1
   
    def return_pattern_width(self):
        return max(self.V[:, 0]) - min(self.V[:, 0])

    def return_pattern_height(self):
        return max(self.V[:, 1]) - min(self.V[:, 1])
    



    def plot(self, location=(0,0)):
        
        initialize_pattern_collection(self.coll_name, self.stroke_coll_name)

        
        construct_object_from_mesh_to_scene(self.V, self.F, self.pattern_name, self.coll_name, self.E)

        mesh = bpy.data.objects[self.pattern_name]
        
        mesh.scale = (1, 1, 1)
        mesh.location = (location[0], location[1], 0)
        mesh.show_axis = False
        mesh.show_wire = True
        mesh.display_type = 'WIRE'
        select_one_object(mesh)
        
        text_loc = (location[0] + min(self.V[:,0]), \
                    location[1] + min(self.V[:,1]) + self.return_pattern_height() + LAYOUT_TEXT, \
                    0)
        # add annotation to full pattern
        add_text_to_scene(body=self.annotation_text, 
                          location=text_loc, 
                          scale=(1,1,1),
                          obj_name=self.pattern_name+"_annotation",
                          coll_name=self.coll_name)
       
        # visualize all stitching lines
        if len(self.stitching_points_line_id) > 0:
          for lid in range(max(self.stitching_points_line_id)+1):
            
            # cannot use the position from the V, since the mesh is translated
#            _, pids = self.get_pts_in_stitching_line(lid)
##            print(pids)
#            vtxID = self.stitching_points_vtx_id[pids]
#            
            vtxID = self.get_vid_in_stitching_line(lid)
            
            pos = get_vtx_pos(mesh, np.array(vtxID))
            
            # draw the stitching lines in the world coordinate
            draw_stitching_line(pos, col_blue, "stitching_line_" + str(lid), STROKE_SIZE, self.stroke_coll_name)
#            add_stroke_to_gpencil(pos, col_blue, "FSP_StitchingLines", STROKE_SIZE)

        if len(bpy.types.Scene.solver_data.hex_centers) > 0:
          centers = bpy.types.Scene.solver_data.hex_centers
          obj = bpy.data.objects[MESH_NAME_FSP]
          mesh = obj.data
          for edge in mesh.edges:
            if edge.vertices[0] not in centers and edge.vertices[1] not in centers:
              edge.bevel_weight = 0.7
            
            
    
    def info(self):
        print('\n------------------- Info: Full Smocking Pattern -------------------')
        print('No. vertices: ' + str(len(self.V)))
        print('No. faces: ' + str(len(self.F)))
        print('No. stitching lines: ' + str(max(self.stitching_points_line_id)+1))
        print('No. unit patches: ' + str(max(self.stitching_points_patch_id)+1))
        print(self.stitching_points_patch_id)
        print('-------------------------------------------------------------------\n')

# ========================================================================
#                      Core functions for smocking pattern
# ========================================================================

class CreateClothSimOperator(Operator):
  bl_idname = "object.create_cloth_sim"
  bl_label = "Creates a cloth simulator for the smocked mesh."
  
  def execute(self, context):
    dt = bpy.types.Scene.solver_data
    fsp = dt.full_smocking_pattern
    sg = dt.smocked_graph
    obj = cloth_sim.create_cloth_mesh(fsp, sg)
    cloth_sim.add_cloth_sim(obj)
    return {'FINISHED'}

class DirectArapOperator(Operator):
  bl_idname = "object.direct_arap_operator"
  bl_label = "Creates a cloth simulator for the smocked mesh."
  
  def execute(self, context):
    dt = bpy.types.Scene.solver_data
    fsp = dt.full_smocking_pattern
    sg = dt.smocked_graph

    # global g_V, g_F, g_mesh, g_iters, g_arap, g_constraints
    obj, V3D, g_F, g_arap = arap_stitching_lines.create_direct_arap_mesh(fsp, sg)
    g_mesh = obj.data
    g_V = V3D.T
    # g_V = g_arap(None, 0, V3D)
    g_iters = 100
    g_constraints = None
    dt.arap_data = [g_V, g_F, g_mesh, g_iters, g_arap, g_constraints]
    return {'FINISHED'}

class RealtimeArapOperator(bpy.types.Operator):
    """Show arap iterations"""
    bl_idname = "arap.realtime_operator"
    bl_label = "Realtime arap operator"

    _timer = None

    def modal(self, context, event):
        if event.type in {'ESC'}:
            self.cancel(context)
            return {'CANCELLED'}

        if event.type == 'TIMER':
            dt = bpy.types.Scene.solver_data
            g_V, _, g_mesh, g_iters, g_arap, g_constraints = dt.arap_data
            wm = context.window_manager
            wm.event_timer_remove(self._timer)         
            g_V = g_arap(g_constraints, 1, g_V.T)
            g_mesh.vertices.foreach_set("co", g_V.T.reshape(g_V.size))
            g_mesh.update()
            g_iters -= 1
            dt.arap_data[0] = g_V
            dt.arap_data[3] = g_iters
            if (g_iters == 0):
              return {'FINISHED'}
            self._timer = wm.event_timer_add(0.01, window=context.window)

        return {'PASS_THROUGH'}

    def execute(self, context):
        wm = context.window_manager
        dt = bpy.types.Scene.solver_data
        dt.arap_data[3] = context.scene.arap_num_iters
        self._timer = wm.event_timer_add(0.01, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        wm = context.window_manager
        wm.event_timer_remove(self._timer)

def prepare_cylinder_arap(V, F, pattern_arap, fsp, sg, weight = 1):
  anchor_ids = pattern_arap.anchors
  nl = fsp.num_stitching_lines()
  underlay_anchors = [anchor_ids[i] for i in range(len(anchor_ids)) if sg.dict_sp2sg[i] < nl]
  num_underlay = len(underlay_anchors)
  underlay_locations = V[underlay_anchors]
  bbox_min, bbox_max = (np.amin(underlay_locations, axis=0), np.amax(underlay_locations, axis=0))
  # The height and circumference of the cylinder. 
  l = bbox_max - bbox_min
  center = (bbox_max + bbox_min) / 2
  # Fold to a cylinder along the x direction.
  r = l[0] / (2 * np.pi)

  cyl_uv, cyl_verts = wrapping.create_cylinder(l)
  constraints = wrapping.get_constraints_from_param(underlay_locations, underlay_anchors,
  cyl_uv, cyl_verts)

  # # Shift up by r.
  # underlay_locations = np.concatenate([underlay_locations[:,0:2], r * np.ones([num_underlay, 1])], axis=1)
  # underlay_locations[:, 0] -= center[0]
  # # Rotate along the y axis.
  # new_locations = []
  # for loc in underlay_locations:
  #   x,y,z = loc
  #   theta = -((x) / (l[0]*0.8)) * np.pi
  #   new_locations.append([x * np.cos(theta) - z * np.sin(theta), y,
  #                         x * np.sin(theta) + z * np.cos(theta)])
  # new_locations = np.array(new_locations)
  # new_locations[:, 0] += center[0]

  # New constraints for underlay.
  # constraints = {underlay_anchors[i] : new_locations[i] for i in range(num_underlay)}


  cyl_arap = arap.ARAP(V.transpose(), F, list(constraints.keys()), weight)
  return cyl_arap, constraints


def prepare_arap(x, fsp, sg, weight = 1):
  bbox_min, bbox_max = (np.amin(fsp.V, axis=0), np.amax(fsp.V, axis=0))
  margin_x, margin_y = (0.5, 0.5)
  grid_size = [int((bbox_max[0] - bbox_min[0] + 2*margin_x) / 0.15),
               int((bbox_max[1] - bbox_min[1] + 2*margin_y) / 0.15)]
  # Create grid.
  [gx, gy]= np.meshgrid( \
    np.linspace(bbox_min[0] - margin_x, bbox_max[0] + margin_x, grid_size[0]),
    np.linspace(bbox_min[1] - margin_y, bbox_max[1] + margin_y, grid_size[1]))
  
  # Create graph from grid. 
  F, V, _ = extract_graph_from_meshgrid(gx, gy, True)
  # Quad -> triangle mesh.
  F = np.array(list(itertools.chain.from_iterable(\
      [(f[[0,1,2]], f[[2,3,0]]) for f in F])))

  # TODO: Replace with a faster version?
  # Find the closest vertex in the grid to each of the vertices in the coarse one.
  vid = [np.argmin(np.linalg.norm(V - p, axis=1)) for p in fsp.V]
  constraints = {vid[i]: x[sg.dict_sp2sg[i]] for i in range(len(vid))}
  # Shift by epsilon.
  # for l in range(len(fsp.stitching_points_line_id)):
  #   vtx = fsp.get_vid_in_stitching_line(l)
  #   for i in range(1, len(vtx)):
  #     eps = np.concatenate([fsp.V[vtx[i], :] - fsp.V[vtx[i - 1], :], [0]]) 
  #     constraints[vid[vtx[i]]] = constraints[vid[vtx[i]]] + eps*1e-3

  # Add z coordinate.
  V3D = np.concatenate((V, np.zeros([V.shape[0], 1])), axis=1)
  grid_arap = arap.ARAP(V3D.transpose(), F, vid, weight)
  return grid_arap, constraints, F, V

def arap_simulate_smocked_design(x, fsp, sg):
  grid_arap, constraints, F, V = prepare_arap(x, fsp, sg)
  res = grid_arap(constraints, 10)
  return [res.transpose(), F, V]


def sort_edge(edges):
    # return the unique edges
    e_sort = [np.array([min(e), max(e)]) for e in edges]
    e_unique = np.unique(e_sort, axis = 0)
    e_unique = np.delete(e_unique, np.where(e_unique[:,0] == e_unique[:, 1]), axis=0)
    return e_unique



def create_grid(num_x, num_y):

    # note num_x/num_y can be float :/
#    xx = range(num_x+1) # [0, ..., num_x]
#    yy = range(num_y+1) # [0, ..., num_y]    
    
    xx = range(int(np.ceil(num_x)) + 1)
    yy = range(int(np.ceil(num_y)) + 1)
    
    gx, gy = np.meshgrid(xx, yy)

    return gx, gy





def add_margin_to_grid(x_ticks, y_ticks,
                       m_left=0, m_right=0, m_top=0, m_bottom=0):
    
    min_x = min(x_ticks)
    max_x = max(x_ticks)
    min_y = min(y_ticks)
    max_y = max(y_ticks)
    
#    print(min_x, max_x, min_y, max_y)
    
    xx = [-m_left + min_x] + list(np.unique(x_ticks)) + [max_x + m_right]
    yy = [-m_bottom + min_y] + list(np.unique(y_ticks)) + [max_y + m_top]
    
    gx, gy = np.meshgrid(np.unique(xx), np.unique(yy))

    return gx, gy

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
    
    # create the grid edges: array
    E = [f[[0,1]] for f in F] + [f[[1,2]] for f in F] + [f[[2,3]] for f in F] + [f[[0,3]] for f in F]
        
    if if_add_diag: # also add the grid diagonal edges
#        print('Add diagonal edges')
        E = E + [f[[0,2]] for f in F] + [f[[1,3]] for f in F]
        
    E = sort_edge(E)
    

    
    return F, V, E




def tile_unit_smocking_pattern_regular(usp, num_x, num_y, shift_x, shift_y):
    
    base_x = usp.base_x
    base_y = usp.base_y
        
    len_x = num_x * base_x - shift_x
    len_y = num_y * base_y - shift_y
        
    # tile the stitching lines from usp
    
    all_sp = [] # all stitching points
    all_sp_lid = [] # the stiching line ID
    all_sp_pid = [] # the stitching patch ID
    
    # extract each stitching lines from the unit pattern
    unit_lines = []
    for line_id in range(len(usp.stitching_lines)):
        pos = usp.get_pts_in_stitching_line(line_id)
        unit_lines.append(pos)    
    
    pid = 0
    lid = 0
    
    for ix in range( int(np.ceil(num_x)) ):
        for iy in range( int(np.ceil(num_y))):
            # [ix, iy] patch
            # shift the unit stitching lines to the current patch
            trans = np.array([ix*base_x, iy*base_y, 0]) - np.array([shift_x, shift_y, 0])
            for line in unit_lines:
                # add the translated stitching points
                new_line = line + trans
                
                # check if the new stitching line is valid, i.e., in the range of [0, len_x]
                if np.min(new_line[:, 0]) < 0 or np.min(new_line[:, 1]) < 0 or np.max(new_line[:, 0]) > len_x or np.max(new_line[:, 1]) > len_y:
                    is_valid = False
                else:
                    is_valid = True        
                
                if is_valid:
                    
                    all_sp += new_line.tolist()
                    
                    # save the line-id of the current line
                    # need to repeat it for each point in this line
                    num = len(line)
                    all_sp_lid += np.repeat(lid, num).tolist()
                    all_sp_pid += np.repeat(pid, num).tolist()
                    
                    lid += 1
            
            pid += 1
    return  all_sp, all_sp_lid, all_sp_pid, len_x, len_y




def combine_two_smocking_patterns(fsp1, fsp2, axis, dist, shift):

    if axis == 'x':
        # [P1, P2]
        p2_trans = [fsp1.return_pattern_width() + dist, shift]
        ind = 0

    elif axis == 'y':
        # [P1
        #  P2]
        p2_trans = [shift, fsp1.return_pattern_height() + dist]
        ind = 1
    

    # if fsp2 has some margin, we need to thift the margin before combining two patterns
    p2_trans[ind] -= min(fsp2.V[:, ind])

    # we translate the vtx and stitching points in P2 by p2_trans
    # then merge them into a singe pattern
    all_sp1 = fsp1.stitching_points
    all_sp2 = fsp2.stitching_points + p2_trans[0:2]

    
    all_sp = np.concatenate((all_sp1, all_sp2))
    all_sp_lid = np.concatenate( (fsp1.stitching_points_line_id, fsp2.stitching_points_line_id + fsp1.num_stitching_lines() ) )

    all_V = np.concatenate( (fsp1.V, fsp2.V + p2_trans) )

    gx, gy = np.meshgrid(np.unique(all_V[:,0]), np.unique(all_V[:,1]))

    F, V, E = extract_graph_from_meshgrid(gx, gy, True)
    
    # create the combined pattern
    fsp = SmockingPattern(V, F, E,
         all_sp,
         all_sp_lid,
         [],
         [],
         'combined_P1_P2', 
         COLL_NAME_FSP_TMP,
         COLL_NAME_FSP_TMP_SL,
         'P1 + P2 combined')

    return fsp    




def deform_regular_into_radial_grid(V, ratio):

    x_ticks = np.unique(V[:,0]) # use this to determine the angle
    y_ticks = np.unique(V[:,1]) # use this to determine the radius



    theta = (x_ticks - min(x_ticks))/(max(x_ticks)-min(x_ticks)) *2*np.pi

    angle_dict = {}
    for x_coord, x_theta in zip(x_ticks, theta):
        angle_dict[x_coord] = x_theta



    width = max(x_ticks) - min(x_ticks)

    radius_dict = {}
    radius_dict[min(y_ticks)] = width / (2*np.pi)

    for y1, y2 in zip(y_ticks[:-1], y_ticks[1:]):
        r1 = radius_dict[y1]

        scale = 2*np.pi*r1 / width
        
        r2 = r1 + (y2 - y1)*scale*ratio
        
        radius_dict[y2] = r2

    V_new = []
    # update the vertex positions
    for x, y in zip(V[:,0], V[:, 1]):
        V_new.append([radius_dict[y]*np.cos( angle_dict[x] ), radius_dict[y]*np.sin( angle_dict[x] )] )

    
    V_new = np.array(V_new)

    return V_new



def deform_fsp_into_radial_grid(fsp, ratio):

    V = fsp.V[:, 0:2]
    V_new = deform_regular_into_radial_grid(V, ratio)

    # remove the duplicated vertices
    # x_min and x_max are merged together
    vid_min = np.array(find_index_in_list(V[:, 0], min(V[:, 0])))
    vid_max = np.array(find_index_in_list(V[:, 0], max(V[:, 0])))

    if len(vid_min) != len(vid_max):
        if_merge = False
    else: 
        ind1 = np.argsort(V[vid_min, 1])
        ind2 = np.argsort(V[vid_max, 1])

        vid_min = vid_min[ind1]
        vid_max = vid_max[ind2]

        v1 = V_new[vid_min, :]
        v2 = V_new[vid_max, :]

        err = np.linalg.norm(v1 - v2)
        
        if err > 1e-6:
            if_merge = False
        else:
            if_merge = True
    
    if not if_merge:
        print('We cannot transform the grid to radial grid, it is already distorted')
    else:
        nv = V.shape[0]
        # replace vid_max by vid_min
        corres = np.empty((nv, 2), 'int')
        corres[:, 0] = np.array(range(0,nv))
        corres[vid_min, 1] = np.array(range(0,len(vid_min)))
        corres[vid_max, 1] = corres[vid_min, 1]
    
        vid_rest = np.setdiff1d(np.array(range(0,nv)), np.concatenate((vid_min, vid_max)))

        corres[vid_rest, 1] = np.array(range(0,len(vid_rest))) + len(vid_min)

        V_new = np.concatenate((V_new[vid_min, :], V_new[vid_rest, :]))


        # i.e., we map corres[:, 0] to corres[:, 1]
        F_new = []
        for f in fsp.F:
            F_new.append(np.array(corres[f, 1]))

        E_new = []    
        for e in fsp.E:
            E_new.append(np.array(corres[e, 1]))

        E_new = np.array(sort_edge(E_new))


        all_sp = V_new[corres[fsp.stitching_points_vtx_id, 1], :]
 
        fsp_new = SmockingPattern(V_new, F_new, E_new,
                                  all_sp,
                                  fsp.stitching_points_line_id,
                                  [],
                                  [],
                                  'RadialPattern', 
                                  COLL_NAME_FSP_TMP,
                                  COLL_NAME_FSP_TMP_SL,
                                  'Pattern in Radial Grid')

        return fsp_new




def extract_smocked_graph_from_full_smocking_pattern(fsp, init_type = 'center'):
    fsp_vid_underlay = fsp.stitching_points_vtx_id
    fsp_vid_pleat = np.setdiff1d(range(fsp.nv), fsp_vid_underlay)

    # use dictionary to record the vertex correspondences between the 
    # smocked graph (sg) and the smocking pattern (sp)
    dict_sp2sg = {}
    dict_sg2sp = {}

    nl = fsp.num_stitching_lines()
    
    # each stiching line in SP becomes a single node in SG
    for lid in range(nl):
        vtxID = fsp.get_vid_in_stitching_line(lid)
        dict_sg2sp[lid] = vtxID

        for vid in vtxID:
            dict_sp2sg[vid] = lid

    count = nl
    for vid in fsp_vid_pleat:
        dict_sp2sg[vid] = count
        dict_sg2sp[count] = [vid]
        count += 1
    
    # by construction we know
    # the first nl vertices are underlay vertices
    vid_underlay = np.array(range(nl))
    # the rest are pleat vertices
    vid_pleat = np.array(range(nl, len(dict_sg2sp)))


    # vtx pos of the smocked graph
    V = fsp.V
    V_sg = np.zeros((len(dict_sg2sp),3))

    for vid in range(len(dict_sg2sp)):
        vtx = dict_sg2sp[vid]
        if len(vtx) > 1:
            if init_type == 'center':
                coord = np.mean(V[vtx, :], axis=0)
            else:
                print('TODO: use random initialization')
        else:
            coord = V[vtx, :]
        V_sg[vid, 0:2] = coord    
        

    # find the edges
    E = fsp.E
    E_sg = []
    for e in E:
        E_sg.append([dict_sp2sg[e[0]], dict_sp2sg[e[1]]])

    E_sg = sort_edge(E_sg)

    # find the faces - for visualization only
    F = fsp.F
    F_sg = []
    for f in F:
        face = []
        for vid in f:
            face.append(dict_sp2sg[vid])
        F_sg.append(np.unique(np.array(face)))
    
    # category the edges
    tmp = np.array(E_sg[:,0] < nl).astype(int) + np.array(E_sg[:, 1] < nl).astype(int)
    eid_underlay = np.array(find_index_in_list(tmp, 2))
    eid_pleat = np.setdiff1d(range(len(E_sg)), eid_underlay)

    return V_sg, E_sg, F_sg, dict_sg2sp, dict_sp2sg, vid_underlay, vid_pleat, eid_underlay, eid_pleat



# TODO: faster solution? now is returning rows with zero L1 distance
def setdiffnd(A, B):
    # A, B: n-dim array
    # set difference between A and B by rows

    # find the row ID with zero L1 distance
    idx = np.where(abs((A[:,np.newaxis,:] - B)).sum(axis=2)==0)

    # A[idx[0]] is equivalent to B[idx[1]]
    idx_diff = np.setdiff1d(range(len(A)), idx[0])

    return idx_diff, idx




def nchoosek(ls_in, n=1):
    c = []
    for i in itertools.combinations(ls_in, n):
        c.append(list(i))
    return np.array(c)



def get_mat_entry(M, I, J):
    res = []
    if len(I) == len(J):
        for i, j in zip(I, J):
            res.append(M[int(i), int(j)])
    else:
        print('Error: the indeices are not consistent')

    return np.array(res)
    



    
# ========================================================================
#                          Utility Functions
# ========================================================================

def delete_everything():
  for coll in bpy.data.collections:
    for item in coll.objects:
      bpy.data.objects.remove(item)
    bpy.data.collections.remove(coll)
  for m in bpy.data.meshes:
    bpy.data.meshes.remove(m)
  for l in bpy.data.lights:
    bpy.data.lights.remove(l)
  for o in bpy.data.objects:
    bpy.data.objects.remove(o)
  bpy.data.orphans_purge()
  bpy.data.orphans_purge()
  bpy.data.orphans_purge()
  bpy.context.window_manager.fsp_drawing_started = False

def delete_all_collections():    
    for coll in bpy.data.collections:
        bpy.data.collections.remove(coll)

def clean_objects_in_collection(coll_name):
    coll = bpy.data.collections[coll_name]
    for item in coll.objects:
        bpy.data.objects.remove(item)    

def delete_all_gpencil_objects():
    for obj in bpy.data.objects:
        if obj.type == 'GPENCIL':
            bpy.data.objects.remove(obj)




def delete_all_objects() -> None:
    for item in bpy.data.objects:
        bpy.data.objects.remove(item)




def clean_one_object(obj_name):
    for item in bpy.data.objects:
        if item.name  == obj_name:
            bpy.data.objects.remove(item)



def select_one_object(obj):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)


        
def deselect_all_vert_in_mesh(obj):
    bm = bmesh.from_edit_mesh(obj.data)     
    for v in bm.verts:
        v.select = False



def find_index_in_list(my_list, my_val):
    return [i for i, x in enumerate(my_list) if x == my_val]

    

def add_text_to_scene(body="test",
                      location=(0,0,0),
                      scale=(1,1,1),
                      obj_name="font_obj",
                      coll_name=COLL_NAME_FSP):
                          
    font_curve = bpy.data.curves.new(type="FONT", name="Font Curve")
    font_curve.body = body
    font_obj = bpy.data.objects.new(name=obj_name, object_data=font_curve)
    font_obj.location = location
    font_obj.scale = scale
#    bpy.context.scene.collection.children[coll_name].objects.link(font_obj)
    bpy.data.collections[coll_name].objects.link(font_obj)



def get_curr_vtx_pos(mesh, vid1):
    p1 = mesh.matrix_world.to_3x3() @ mesh.data.vertices[vid1].co
    p1[0] += mesh.matrix_world[0][3]
    p1[1] += mesh.matrix_world[1][3]
    p1[2] += mesh.matrix_world[2][3]
    return list(p1)

def get_translation_of_mesh(mesh_name):
    mesh = bpy.data.objects[mesh_name]
    return [mesh.matrix_world[0][3], mesh.matrix_world[1][3], mesh.matrix_world[2][3]]




def get_vtx_pos(mesh, vids):
    p = [];
    for vid in vids:
        p.append(get_curr_vtx_pos(mesh, vid))
    return p



def write_fsp_to_obj(fsp, filepath):

    with open(filepath, 'w') as f:
        f.write("# OBJ file\n")
        # write vertices - z-val = 0
        for v_id in range(len(fsp.V)):
            f.write("v %.4f %.4f %.4f\n" % (fsp.V[v_id, 0], fsp.V[v_id, 1], 0))
            
        # write faces
        for f_id in range(len(fsp.F)):
            f.write("f")
            p = fsp.F[f_id]
            for v_id in range(len(p)):
                f.write(" %d" % (p[v_id] + 1))
            f.write("\n")

        # write edges
        for e_id in range(len(fsp.E)):
            f.write("e %d %d\n" % (fsp.E[e_id][0] + 1, fsp.E[e_id][1] + 1))
    
        # add stiching lines to the object
        for lid in range(fsp.num_stitching_lines()):
            vtxID = fsp.get_vid_in_stitching_line(lid)
            
            for ii in range(len(vtxID)-1):
                # f.write('l ' + str(vtxID[ii]+1) + ' ' + str(vtxID[ii+1]+1) + '\n')
                f.write('l %d %d\n' % (vtxID[ii] + 1 , vtxID[ii+1] + 1))
                

def read_obj_to_fsp(file_name, 
                    pattern_name = "FllPattern", 
                    coll_name=COLL_NAME_FSP,
                    stroke_coll_name = COLL_NAME_FSP_SL,
                    annotation_text="Full Smocking Pattern"):       
    V, F, E, all_sp_lid, all_sp_vid = [], [], [], [], []

    file = open(file_name, 'r')
    Lines = file.readlines()
    line_end = None
    lid = -1

    for line in Lines:
        elems = line.split()
        if elems:
            if elems[0] == "v":
                vtx = []
                for ii in range(1, len(elems)):
                    vtx.append(float(elems[ii]))
                V.append(vtx)

            elif elems[0] == "f":
                face = []
                for jj in range(1, len(elems)):
                    face.append(int(elems[jj]) - 1)
                F.append(np.array(face))
            elif elems[0] == "e":
                edge = []
                for kk in range(1, len(elems)):
                    edge.append(int(elems[kk]) - 1)
                E.append(edge)

            elif elems[0] == "l":
                if int(elems[1]) != line_end:
                    lid += 1 # not the same stitching line

                    for ii in range(2): # each line has two vtx
                        all_sp_vid.append(int(elems[ii+1]) - 1)
                        all_sp_lid.append(lid)                       

                else: # same stitching line: only same the second point
                    all_sp_vid.append(int(elems[2]) - 1)
                    all_sp_lid.append(lid)

                # update the line_end    
                line_end = int(elems[2]) 


    file.close()

    E = np.array(E)
    V = np.array(V)
    all_sp = V[all_sp_vid]

    fsp = SmockingPattern(V[:,0:2], F, E, 
                          all_sp, all_sp_lid, [], all_sp_vid,
                          pattern_name, coll_name, stroke_coll_name, annotation_text)

    return fsp
    
    
    
def write_mesh_to_obj(mesh_name, save_dir, save_name):
    mesh = bpy.data.objects[mesh_name]
    select_one_object(mesh)
    bpy.ops.export_scene.obj(filepath= save_dir + save_name, 
                         check_existing=True, 
                         filter_glob='*.obj;*.mtl', 
                         use_selection=True, 
                         use_animation=False, 
                         use_mesh_modifiers=True, 
                         use_edges=True, 
                         use_smooth_groups=False, 
                         use_smooth_groups_bitflags=False, 
                         use_normals=True, 
                         use_uvs=True, 
                         use_materials=True, 
                         use_triangles=False, 
                         use_nurbs=False, 
                         use_vertex_groups=False, 
                         use_blen_objects=True, 
                         group_by_object=False, 
                         group_by_material=False, 
                         keep_vertex_order=False, 
                         global_scale=1.0, 
                         path_mode='AUTO', 
                         axis_forward='-Z', 
                         axis_up='Y')

# ---------------------BEGIN: write/read unit-smocking-pattern ------------------------

def get_usp_from_saved_stitching_lines(all_sl, mesh, base_x, base_y):
    all_sp = []
    all_sp_lid = []
    
    lid = 0
    # write the stiching lines
    for line in all_sl:
        
        for vid in line:
            pos = list(mesh.data.vertices[vid].co)
            all_sp.append(pos)
            all_sp_lid.append(lid)
    
        lid += 1
    
    usp = UnitSmockingPattern(base_x, base_y, all_sl, all_sp, all_sp_lid)
              
    return usp


def write_usp(file_name, usp):
    file = open(file_name, 'w')
    file.write('gridX\t' + str(usp.base_x) + "\n")
    file.write('gridY\t' + str(usp.base_y) + "\n")
    
    # write the stitching lint point IDs - for blender
    for line in usp.stitching_lines:
        file.write('sl\t')
        for vid in line:
            file.write(str(vid)+"\t")
        file.write("\n")
    
    for ii in range(len(usp.stitching_points_line_id)):
        pos = usp.stitching_points[ii]
        
        lid = usp.stitching_points_line_id[ii]
        
        file.write('sp \t' + str(lid) + '\t')
            
        for val in pos:
            file.write(str(val)+"\t")
        file.write("\n")        
        
    file.close()

    
    
    
def read_usp(file_name):
    file = open(file_name, 'r')
    Lines = file.readlines()
    
    all_sl = [] # the stitching line (vtx ID)
    all_sp = [] # all stitching points (positions)
    all_sp_lid = [] # the stitching lineID of each point
    
    for line in Lines:
        elems = line.split()
        if elems:
            if elems[0] == "gridX":
                base_x = int(elems[1])
            
            elif elems[0] =="gridY":
                base_y = int(elems[1])
            
            elif elems[0] == "sl":
                sl = []
                for ii in range(1, len(elems)):
                    sl.append(int(elems[ii]))
                
                all_sl.append(sl)
            
            elif elems[0] == "sp":
                all_sp_lid.append(int(elems[1]))
                all_sp.append([ float(elems[2]), float(elems[3]), float(elems[4]) ])
            
            else:    
                pass
    
    usp = UnitSmockingPattern(base_x, base_y, all_sl, all_sp, all_sp_lid)
        
    return usp

# ---------------------END: write/read unit-smocking-pattern ------------------------


# ---------------------BEGIN: add strokes via gpencil ------------------------

# Drawing with Gpencil:
# Reference: 
# https://gist.github.com/blender8r/4688b3f05640737236c856bc7df47bee
# https://www.youtube.com/watch?v=csQNmnc5xQg

def add_stroke_to_gpencil(pts, col, gpencil_name, line_width=12):
    gpencil = bpy.data.grease_pencils[gpencil_name]
        
    gp_layer = gpencil.layers.new("lines")

    gp_frame = gp_layer.frames.new(bpy.context.scene.frame_current)


    gp_stroke = gp_frame.strokes.new()
    gp_stroke.line_width = line_width
    gp_stroke.start_cap_mode = 'ROUND'
    gp_stroke.end_cap_mode = 'ROUND'
    gp_stroke.use_cyclic = False
    gp_stroke.points.add(len(pts))

    for item, value in enumerate(pts):
        gp_stroke.points[item].co = value
        gp_stroke.points[item].pressure = 10
        
    mat = bpy.data.materials.new(name="Black")
    bpy.data.materials.create_gpencil_data(mat)
    gpencil.materials.append(mat)
    mat.grease_pencil.show_fill = False
#    mat.grease_pencil.fill_color = (1.0, 0.0, 1.0, 1.0)
    mat.grease_pencil.color = (col[0], col[1], col[2], 1.0)
    
    if len(pts) > 2:
        gp_stroke.points[0].pressure = 2
        gp_stroke.points[-1].pressure = 2
    
    

def create_line_stroke_from_gpencil(name="GPencil", line_width=12, coll_name=COLL_NAME_FSP):
    gpencil_data = bpy.data.grease_pencils.new(name)
    gpencil = bpy.data.objects.new(gpencil_data.name, gpencil_data)
#    bpy.context.collection.objects.link(gpencil)
#    bpy.context.scene.collection.children[coll_name].objects.link(gpencil)
    bpy.data.collections[coll_name].objects.link(gpencil)
    gp_layer = gpencil_data.layers.new("lines")

    gp_frame = gp_layer.frames.new(bpy.context.scene.frame_current)

    gp_stroke = gp_frame.strokes.new()
    gp_stroke.line_width = line_width
    gp_stroke.start_cap_mode = 'ROUND'
    gp_stroke.end_cap_mode = 'ROUND'
    gp_stroke.use_cyclic = False

    return gpencil, gp_stroke
    

def draw_stitching_line(pts, col, name="stitching_line", line_width=12, coll_name=COLL_NAME_FSP): 
    gpencil, gp_stroke = create_line_stroke_from_gpencil(name, line_width, coll_name)
    
    if len(pts) == 2: # we add a midpoint to make the drawing looks nicer
        gp_stroke.points.add(len(pts)+1)
        
        gp_stroke.points[0].co = pts[0]
        gp_stroke.points[0].pressure = 2
        gp_stroke.points[1].co = (np.array(pts[0]) + np.array(pts[1]))*0.5
        gp_stroke.points[1].pressure = 10

        gp_stroke.points[2].co = pts[1]
        gp_stroke.points[2].pressure = 2

    else:
        gp_stroke.points.add(len(pts))
        
        for item, value in enumerate(pts):
            gp_stroke.points[item].co = value
            gp_stroke.points[item].pressure = 10
    


        
    mat = bpy.data.materials.new(name="Black")
    bpy.data.materials.create_gpencil_data(mat)
    gpencil.data.materials.append(mat)
    mat.grease_pencil.show_fill = False
#    mat.grease_pencil.fill_color = (1.0, 0.0, 1.0, 1.0)
    mat.grease_pencil.color = (col[0], col[1], col[2], 1.0)
    
    if len(pts) > 2:
        gp_stroke.points[0].pressure = 2
        gp_stroke.points[-1].pressure = 2
#       gp_stroke.points[0].vertex_color = (1.0, 0.0, 0.0, 1.0)
#       gp_stroke.points[-1].vertex_color = (0.0, 1.0, 0.0, 1.0)


def draw_saved_stitching_lines(context, coll_name=COLL_NAME_FSP):
    props = bpy.context.scene.sl_props
    print(props.savedStitchingLines)
    for i in range(len(props.savedStitchingLines)):
        vids = props.savedStitchingLines[i]
        obj = bpy.data.objects[MESH_NAME_USP]
        pts = get_vtx_pos(obj, vids)
        draw_stitching_line(pts, props.colSaved, "stitching_line_" + str(i), STROKE_SIZE, coll_name)
        
        
            
# ---------------------END: add strokes via gpencil ------------------------

            
def construct_object_from_mesh_to_scene(V, F, mesh_name, coll_name=COLL_NAME_FSP, E = []):
    # input: V nv-by-2(3) array, F list of array
    
    # convert F into a list of list
    faces = F
    # convert V into a list of Vector()
    verts = [Vector((v[0], v[1], v[2] if len(v) > 2 else 0)) for v in V]
    
    # create mesh in blender
    mesh = bpy.data.meshes.new(mesh_name)
    mesh.from_pydata(verts, E, faces)
    mesh.update(calc_edges=False) # we use our own edgeID
    object = bpy.data.objects.new(mesh_name, mesh)
    # link the object to the scene
    bpy.context.scene.collection.children[coll_name].objects.link(object)
    

def show_mesh(mesh, scale=(1,1,1), location=(0,0,0)):
    mesh.scale = scale
    mesh.location = location
    mesh.show_axis = False
    mesh.show_wire = True
    mesh.display_type = 'WIRE'
    select_one_object(mesh)



def generate_grid_for_unit_pattern(base_x, base_y, if_add_diag=False):
    # there is some but in blender subdivison
    # instead, manually create a grid
    
    gx, gy = create_grid(base_x, base_y)
    F, V, _ = extract_graph_from_meshgrid(gx, gy, if_add_diag)

    construct_object_from_mesh_to_scene(V, F, MESH_NAME_USP, COLL_NAME_USP)
    
    mesh = bpy.data.objects[MESH_NAME_USP]

    usp_loc, _ = update_usp_fsp_location()

    show_mesh(mesh,
        scale=(1,1,1),
        location=(usp_loc[0], usp_loc[1], 0))

    return mesh, F, V
    

def generate_tiled_grid_for_full_pattern(len_x, len_y, if_add_diag=True):
    # create the full grid with size len_x by len_y
    gx, gy = create_grid(len_x, len_y)
    
    F, V, E = extract_graph_from_meshgrid(gx, gy, if_add_diag)
    
    construct_object_from_mesh_to_scene(V, F, MESH_NAME_FSP, COLL_NAME_FSP)
    
    mesh = bpy.data.objects[MESH_NAME_FSP]

    _, fsp_loc = update_usp_fsp_location()
    
    show_mesh(mesh, 
              scale=(1,1,1), 
              location=(fsp_loc[0], fsp_loc[1], 0))
    
                
    return mesh, F, V, E
    
        

def find_matching_rowID(V, pos):
    # V: a list of vtx positions
    # pos: the query position
    ind = np.where(np.linalg.norm(np.array(V) - np.array(pos[:len(V[0])]), axis=1) < 1e-5)
    # ind = np.where(np.all(np.array(V) == np.array(pos[:len(V[0])]), axis=1))
    return ind


        

def refresh_stitching_lines(fsp):  
    all_sp = []
    all_sp_lid = []
    lid = 0
    
    trans = get_translation_of_mesh(MESH_NAME_FSP) 
    
    # check the remaining stitching lines
    saved_sl_names = []
    for obj in bpy.data.collections[COLL_NAME_FSP_SL].objects:
        saved_sl_names.append(obj.name)
#    print(saved_sl_names)
    
    for sl_name in saved_sl_names:
        gp = bpy.data.grease_pencils[sl_name]
        pts = gp.layers.active.active_frame.strokes[0].points

        for p in pts:
            # p is in the world coordinate
            # need to translate it back to the grid-coordinate
            # grid_location = np.array([p.co[0]-trans[0], p.co[1]-trans[1], p.co[2]-trans[2]])
            grid_location = np.array([p.co[0]-trans[0], p.co[1]-trans[1]])
            if np.any(np.linalg.norm(fsp.V - grid_location, axis=1) < 1e-5):
              all_sp.append(grid_location)
              all_sp_lid.append(lid)
        lid += 1
        

    all_sp_pid = all_sp_lid # now patchID is useless
    
    return all_sp, all_sp_lid, all_sp_pid




    
def initialize_collections():
    # delete_all_objects()
    # delete_all_collections()   
    delete_everything()
    initialize_data()

    initialize_pattern_collection(COLL_NAME_USP, COLL_NAME_USP_SL)
    initialize_pattern_collection(COLL_NAME_FSP, COLL_NAME_FSP_SL)
    initialize_pattern_collection(COLL_NAME_P1, COLL_NAME_P1_SL)
    initialize_pattern_collection(COLL_NAME_P2, COLL_NAME_P2_SL)
    initialize_pattern_collection(COLL_NAME_FSP_TMP, COLL_NAME_FSP_TMP_SL)
    initialize_pattern_collection(COLL_NAME_SG, COLL_NAME_SG_SL)
    

def initialize_data():
    props = bpy.types.Scene.sl_props
    props.runtime_log = []
    dt = bpy.types.Scene.solver_data
    dt.unit_smocking_pattern = []
    dt.full_smocking_pattern = []
    dt.smocked_graph = None
    dt.arap_data = None
    dt.embeded_graph = None
    dt.tmp_fsp1 = []
    dt.tmp_fsp2 = []
    dt.tmp_fsp = []
    dt.hex_centers = []

    for mat in bpy.data.materials:
      bpy.data.materials.remove(mat)
      
    with bpy.data.libraries.load(os.path.join(FILE_PATH, "fabric_material.blend")) as (data_from, data_to):
      data_to.materials = data_from.materials
    
    area = next(area for area in bpy.context.screen.areas if area.type == 'VIEW_3D')
    space = next(space for space in area.spaces if space.type == 'VIEW_3D')
    space.shading.type = 'RENDERED'
    light_data = bpy.data.lights.new(name="sun_data", type='SUN')
    light_data.energy = 0.8
    light_object = bpy.data.objects.new(name="sun", object_data=light_data)
    light_object.rotation_euler.y = 0.2
    light_object.rotation_euler.x = -0.2
    bpy.context.collection.objects.link(light_object)
    bpy.data.scenes[0].world.use_nodes = True
    bpy.data.scenes[0].world.node_tree.nodes["Background"].inputs['Color'].default_value = (0.05, 0.05, 0.05, 1.0)



def print_runtime():

    runtime = bpy.types.Scene.sl_props.runtime_log
    print('\n--------------------------- Runtime Log ---------------------------')
    

    for msg in runtime:
        print(msg)
    print('-------------------------------------------------------------------\n')
            


def initialize_pattern_collection(coll_name, stroke_coll_name):
    if_coll_exist = False

    for coll in bpy.data.collections:
        # if coll_name in coll.name:
        if coll_name == coll.name.split(".")[0]:
            if_coll_exist = True

    if not if_coll_exist:

        my_coll = bpy.data.collections.new(coll_name)
        bpy.context.scene.collection.children.link(my_coll)

        my_coll_strokes = bpy.data.collections.new(stroke_coll_name)
        my_coll.children.link(my_coll_strokes)
    

    for c in [coll_name, stroke_coll_name]:
            clean_objects_in_collection(c)




#-------------------------  update the layout to avoid overlapping


# TODO: make the layout better
def update_tmp_pattern_location():
    dt = bpy.types.Scene.solver_data

    fsp1 = dt.tmp_fsp1  
    fsp2 = dt.tmp_fsp2
    fsp3 = dt.tmp_fsp

    # initialize the location

    # for each pattern the origin is at (0,0), which is not necessarily at the left-bottom corner

    if fsp1 != []:
        fsp1_loc = [-fsp1.return_pattern_width() - min(fsp1.V[:,0]) - LAYOUT_X_SHIFT, -min(fsp1.V[:, 1])]
    else:
        fsp1_loc = [-10,2]

    if fsp2 != []:
        fsp2_loc = [-fsp2.return_pattern_width() - min(fsp2.V[:,0]) - LAYOUT_X_SHIFT, -min(fsp2.V[:, 1])]
    else:
        fsp2_loc = [-10,2]

    if fsp3 != []:
        fsp3_loc = [-fsp3.return_pattern_width() - min(fsp3.V[:,0]) - LAYOUT_X_SHIFT, -min(fsp3.V[:, 1])]
    else:
        fsp3_loc = [-10,2]

    # update the location

    if fsp3 != []:     

        fsp2_loc[1] += fsp3_loc[1] + fsp3.return_pattern_height()  + LAYOUT_Y_SHIFT

        if fsp2 != []:      

            fsp1_loc[1] += fsp2_loc[1] + fsp2.return_pattern_height()  + LAYOUT_Y_SHIFT

        else:

            fsp1_loc[1] += fsp3_loc[1] + fsp3.return_pattern_height()  + LAYOUT_Y_SHIFT

    elif fsp2 != []:

        fsp1_loc[1] += fsp2_loc[1] + fsp2.return_pattern_height()  + LAYOUT_Y_SHIFT



    return fsp1_loc, fsp2_loc, fsp3_loc



def update_usp_fsp_location():
    dt = bpy.types.Scene.solver_data
    usp = dt.unit_smocking_pattern
    fsp = dt.full_smocking_pattern

    usp_loc = [0,5]
    fsp_loc = [0,0]

    if fsp != []:
        fsp_loc[0] -= min(fsp.V[:,0])
        fsp_loc[1] -= min(fsp.V[:,1])
        usp_loc[1] = max(usp_loc[1], fsp.return_pattern_height() + LAYOUT_USP_FSP_SPACE)

    return usp_loc, fsp_loc


def update_smocked_graph_location():
    print('not done yet')




# ========================================================================
#                          Functions for UIs
# ========================================================================

# ------------------------------------------------------------------------
#    Drawing the Unit Pattern using mouse
# ------------------------------------------------------------------------

from bpy.props import IntProperty, FloatProperty

class USP_SelectStitchingPoint(Operator):
    """Draw a stitching line by selecting points in order"""
    bl_idname = "object.modal_operator"
    bl_label = "Simple Modal Operator"
   
    def modal(self, context, event):
        
        props = bpy.context.scene.sl_props
        
        if props.if_user_is_drawing:
        
            if event.type == 'LEFTMOUSE':

#                print('I hit the left mouse')
                
                obj = bpy.data.objects[MESH_NAME_USP]
                bm = bmesh.from_edit_mesh(obj.data)

                for v in bm.verts:
                    if v.select:
                        if v.index not in props.currentDrawing:
                            props.currentDrawing.append(v.index)
#                            print(props.currentDrawing)
        else:
            return{'CANCELLED'}
        
        return {'PASS_THROUGH'}
        

    
    def invoke(self, context, event):
        
        
        delete_all_gpencil_objects()
        
        draw_saved_stitching_lines(context)
        
        props = bpy.context.scene.sl_props
        
        props.if_user_is_drawing = True
        props.if_curr_drawing_is_shown = False
        context.window_manager.usp_drawing_started = True
        
        

        if props.if_user_is_drawing:
            props.currentDrawing = []
            obj = bpy.data.objects[MESH_NAME_USP]
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode = 'EDIT') 
            # enter vertex selection mode
            bpy.context.tool_settings.mesh_select_mode = (True, False, False)
            deselect_all_vert_in_mesh(obj)
            
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            self.report({'WARNING'}, "No active object, could not finish")
            return {'CANCELLED'}

        
class USP_FinishCurrentDrawing(Operator):
    """Finish drawing the current stitching line"""
    bl_idname = "edit.finish_drawing"
    bl_label = "Finish drawing the current stitching line"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        
        bpy.ops.object.mode_set(mode = 'OBJECT')
        
        props = bpy.context.scene.sl_props
        
        if not props.if_curr_drawing_is_shown:
            mesh = bpy.data.objects[MESH_NAME_USP]
            pts = get_vtx_pos(mesh, props.currentDrawing)
            draw_stitching_line(pts, props.colTmp, "stitching_line_tmp", STROKE_SIZE, COLL_NAME_USP_SL)
        
            props.if_curr_drawing_is_shown = True

        props.if_user_is_drawing = False
        context.window_manager.usp_drawing_started = False

        return {'FINISHED'}


class USP_FinishPattern(Operator):
    """Finish creating the unit smocking pattern"""
    bl_idname = "edit.finish_creating_pattern"
    bl_label = "Finish creating the unit smocking pattern"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
            
        props = bpy.context.scene.sl_props
                
        mesh = bpy.data.objects[MESH_NAME_USP]
        
        usp = get_usp_from_saved_stitching_lines(props.savedStitchingLines, 
                                                 mesh, 
                                                 context.scene.base_x,
                                                 context.scene.base_y)

        usp_loc, _ = update_usp_fsp_location()
        usp.plot(usp_loc)
        
        # save the loaded pattern to the scene
        bpy.types.Scene.solver_data.unit_smocking_pattern = usp
        
        return {'FINISHED'}



class USP_SaveCurrentStitchingLine(Operator):
    """Save this stitching line"""
    bl_idname = "edit.save_current_stitching_line"
    bl_label = "Add one stiching line"
    bl_options = {'REGISTER', 'UNDO'}
        
    def execute(self, context):
        props = bpy.context.scene.sl_props
        
        if props.currentDrawing:
            props.savedStitchingLines.append(props.currentDrawing)
        
        props.currentDrawing = []
        
#        print(props.savedStitchingLines)
        
        clean_objects_in_collection(COLL_NAME_USP_SL)
        
        draw_saved_stitching_lines(context, COLL_NAME_USP_SL)
        
        props.if_curr_drawing_is_shown = True
        
        return {'FINISHED'}





class ExportUnitPattern(Operator):
    """Export this unit smocking pattern"""
    bl_idname = "object.export_unit_pattern"
    bl_label = "Export the unit pattern to file"

    
    def execute(self, context):
        props = bpy.types.Scene.sl_props
        save_dir = bpy.path.abspath(context.scene.path_export)
        save_name = context.scene.filename_export
        
        file_name = save_dir + save_name + '.usp'
        
        
        usp = get_usp_from_saved_stitching_lines(props.savedStitchingLines, 
                                                 bpy.data.objects[MESH_NAME_USP], 
                                                 context.scene.base_x,
                                                 context.scene.base_y)
        write_usp(file_name, usp)
        
        return {'FINISHED'}
    



class ImportUnitPattern(Operator):
    """Import an existing unit smocking pattern"""
    bl_idname = "object.import_unit_pattern"
    bl_label = "Import a unit pattern"
    
    def execute(self, context):
        # refresh the drawing of the unit pattern
        # clean usp and fsp, don't touch the tmp collections
        initialize_pattern_collection(COLL_NAME_USP, COLL_NAME_USP_SL)
        initialize_pattern_collection(COLL_NAME_FSP, COLL_NAME_FSP_SL)

        
        file_name = bpy.path.abspath(context.scene.path_import)
        usp = read_usp(file_name)
        
        usp_loc, _ = update_usp_fsp_location()
        usp.plot(usp_loc)
        # save the loaded pattern to the scene
        bpy.types.Scene.solver_data.unit_smocking_pattern = usp
        
        return {'FINISHED'}
        
    



# ------------------------------------------------------------------------
#    Create Unit Grid
# ------------------------------------------------------------------------

class USP_CreateGrid(Operator):
    """Create a grid for specifying unit smocking pattern"""
    bl_idname = "object.create_grid"
    bl_label = "Create Grid"
    bl_options = {'REGISTER', 'UNDO'}

    

    def execute(self, context):
        
        initialize_pattern_collection(COLL_NAME_USP, COLL_NAME_USP_SL)
        initialize_pattern_collection(COLL_NAME_FSP, COLL_NAME_FSP_SL)
        

        base_x = context.scene.base_x
        base_y = context.scene.base_y
        
        generate_grid_for_unit_pattern(base_x, base_y)

        usp_loc, _ = update_usp_fsp_location()


        add_text_to_scene(body="Unit Smocking Pattern", 
                          location=(0, usp_loc[1] + base_y + LAYOUT_TEXT, 0), 
                          scale=(1,1,1),
                          obj_name='grid_annotation',
                          coll_name=COLL_NAME_USP)
                          
        # clear the old drawings
        props = bpy.types.Scene.sl_props
        props.savedStitchingLines = []
        props.currentDrawing = []
        
        return {'FINISHED'}



# ------------------------------------------------------------------------
#    Full Smocking Pattern by Tiling
# ------------------------------------------------------------------------
class HoneycombGrid(Operator):
  """Generate the full smokcing pattern by tiling the specified unit pattern"""
  bl_idname = "object.create_honeycomb_grid"
  bl_label = "Generate Full Smocking Pattern"
  bl_options = {'REGISTER', 'UNDO'}

  def execute(self, context):
    num_x = context.scene.num_x
    num_y = context.scene.num_y
    V, E, centers = honeycomb.generate_hex_grid(num_x, num_y)
    
    F = []
    all_sp = []
    all_sp_lid = []
    all_sp_pid = []
    fsp = SmockingPattern(V, F, E,
                             all_sp, 
                             all_sp_lid,
                             all_sp_pid, [],
                             MESH_NAME_FSP, 
                              COLL_NAME_FSP,
                              COLL_NAME_FSP_SL,
                              "Full Smocking Pattern")
    bpy.types.Scene.solver_data.full_smocking_pattern = fsp
    bpy.types.Scene.solver_data.hex_centers = centers

    usp_loc, fsp_loc = update_usp_fsp_location()

    fsp.plot(fsp_loc)
    return {'FINISHED'}

class FSP_Tile(Operator):
    """Generate the full smokcing pattern by tiling the specified unit pattern"""
    bl_idname = "object.create_full_smocking_pattern"
    bl_label = "Generate Full Smocking Pattern"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        
        props = bpy.types.Scene.sl_props
        dt = bpy.types.Scene.solver_data
        usp = dt.unit_smocking_pattern
        
        
        base_x = usp.base_x
        base_y = usp.base_y
        num_x = context.scene.num_x
        num_y = context.scene.num_y
        shift_x = context.scene.shift_x
        shift_y = context.scene.shift_y
        
        

        
        all_sp, all_sp_lid, all_sp_pid, len_x, len_y = tile_unit_smocking_pattern_regular(usp, num_x, num_y, shift_x, shift_y)
        
        mesh, F, V, E = generate_tiled_grid_for_full_pattern(len_x, len_y, True)
    
        fsp = SmockingPattern(V, F, E,
                             all_sp, 
                             all_sp_lid,
                             all_sp_pid,
                             [],
                             MESH_NAME_FSP, 
                             COLL_NAME_FSP,
                             COLL_NAME_FSP_SL,
                             "Full Smocking Pattern")


        # save the loaded pattern to the scene
        bpy.types.Scene.solver_data.full_smocking_pattern = fsp


        usp_loc, fsp_loc = update_usp_fsp_location()

        fsp.plot(fsp_loc)

        # also update the usp plot
        usp = bpy.types.Scene.solver_data.unit_smocking_pattern
        usp.plot(usp_loc)

        
        
        
    
        return {'FINISHED'}
    

class FSP_AddMargin(Operator):
    bl_idname = "object.full_smocking_pattern_add_margin"
    bl_label = "Add Margin to Current Smocking Pattern"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):

        m_top = context.scene.margin_top
        m_bottom = context.scene.margin_bottom
        m_left = context.scene.margin_left
        m_right = context.scene.margin_right
        
        
        dt = bpy.types.Scene.solver_data
        fsp = dt.full_smocking_pattern
        
        
        V = fsp.V
        
        gx, gy = add_margin_to_grid(np.unique(V[:,0]), np.unique(V[:, 1]),
                                    m_left, m_right, m_top, m_bottom)
    
        F, V, E = extract_graph_from_meshgrid(gx, gy, True)
        
        initialize_pattern_collection(COLL_NAME_FSP_TMP, COLL_NAME_FSP_TMP_SL)

        fsp_new = SmockingPattern(V, F, E,
                                  fsp.stitching_points,
                                  fsp.stitching_points_line_id,
                                  [],
                                  [],
                                  'tmp', 
                                  COLL_NAME_FSP_TMP,
                                  COLL_NAME_FSP_TMP_SL,
                                  'Preview: add margin')
        dt.tmp_fsp = fsp_new
        dt.tmp_fsp.plot((-fsp_new.return_pattern_width()-1, 0, 0))
        
        return {'FINISHED'}

class FSP_Import(Operator):
    bl_idname = "object.full_smocking_pattern_import"
    bl_label = "Import the Smocking Pattern to A Mesh"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):

        dt = bpy.types.Scene.solver_data
        dt.full_smocking_pattern = read_obj_to_fsp(bpy.path.abspath(context.scene.path_import_fullpattern), 
        MESH_NAME_FSP, COLL_NAME_FSP, COLL_NAME_FSP_SL, "Full Smocking Pattern")

        usp_loc, fsp_loc = update_usp_fsp_location()
        # usp.plot(usp_loc)
        dt.full_smocking_pattern.plot(fsp_loc)

        return {'FINISHED'}

class FSP_Export(Operator):
    bl_idname = "object.full_smocking_pattern_export"
    bl_label = "Export the Smocking Pattern to A Mesh"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        
        dt = bpy.types.Scene.solver_data
        
        save_dir = bpy.path.abspath(context.scene.path_export_fullpattern)
        file_name = context.scene.filename_export_fullpattern + context.scene.export_format

        if context.scene.export_format == ".obj":
            
            fsp = dt.full_smocking_pattern
            filepath = save_dir + file_name
            write_fsp_to_obj(fsp, filepath)
        else:
            print('not done yet:/')

        return {'FINISHED'}
    

class FSP_DeleteStitchingLines_start(Operator):
    bl_idname = "object.fsp_delete_stitching_lines_start"
    bl_label = "Delete stitching lines start"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        bpy.ops.object.select_all(action='DESELECT')
        
        # select all stitching lines in the full pattern
        for obj in bpy.data.collections[COLL_NAME_FSP_SL].objects:
            obj.select_set(True)
            
        return {'FINISHED'}


    
    

class FSP_DeleteStitchingLines_done(Operator):
    bl_idname = "object.fsp_delete_stitching_lines_done"
    bl_label = "Delete stitching lines done"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
         
        dt = bpy.types.Scene.solver_data
        fsp = dt.full_smocking_pattern
        
        
        all_sp, all_sp_lid, all_sp_pid = refresh_stitching_lines(fsp)
        
        fsp.update_stitching_lines(all_sp, all_sp_lid, all_sp_pid)

        usp_loc, fsp_loc = update_usp_fsp_location()
        # usp.plot(usp_loc)
        fsp.plot(fsp_loc)
                    
        return {'FINISHED'}




class FSP_AddStitchingLines_draw_start(Operator):
    bl_idname = "object.fsp_add_stitching_lines_draw_start"
    bl_label = "Start drawing a new stitching line"
    bl_options = {'REGISTER', 'UNDO'}
    
    def modal(self, context, event):
        
        props = bpy.context.scene.sl_props
        
        if props.if_user_is_drawing:
        
            if event.type == 'LEFTMOUSE':

                obj = bpy.data.objects[MESH_NAME_FSP]
                bm = bmesh.from_edit_mesh(obj.data)

                for v in bm.verts:
                    if v.select:
                        if v.index not in props.currentDrawing:
                            props.currentDrawing.append(v.index)
#                            print(props.currentDrawing)
        else:
            return{'CANCELLED'}
        
        return {'PASS_THROUGH'}
        

    
    def invoke(self, context, event):
        
        # remove the temporay stitching lines
        props = bpy.context.scene.sl_props
        for obj in bpy.data.collections[COLL_NAME_FSP_SL].objects:
            if 'tmp' in obj.name:
                bpy.data.objects.remove(obj)
         
        props.if_user_is_drawing = True
        props.if_curr_drawing_is_shown = False
        context.window_manager.fsp_drawing_started = True
        

        if props.if_user_is_drawing:
            props.currentDrawing = []
            obj = bpy.data.objects[MESH_NAME_FSP]
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode = 'EDIT') 
            bpy.context.tool_settings.mesh_select_mode = (True, False, False)
            
            deselect_all_vert_in_mesh(obj)
            
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            self.report({'WARNING'}, "No active object, could not finish")
            return {'CANCELLED'}




class FSP_AddStitchingLines_draw_end(Operator):
    bl_idname = "object.fsp_add_stitching_lines_draw_end"
    bl_label = "Finish drawing a new stitching line"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        bpy.ops.object.mode_set(mode = 'OBJECT')
        
        props = bpy.context.scene.sl_props
        
        if not props.if_curr_drawing_is_shown:
            mesh = bpy.data.objects[MESH_NAME_FSP]
            pts = get_vtx_pos(mesh, props.currentDrawing)
            draw_stitching_line(pts, props.colTmp, "stitching_line_tmp", STROKE_SIZE, COLL_NAME_FSP_SL)
        
            props.if_curr_drawing_is_shown = True

        props.if_user_is_drawing = False
        context.window_manager.fsp_drawing_started = False
        return {'FINISHED'}




class FSP_AddStitchingLines_draw_add(Operator):
    bl_idname = "object.fsp_add_stitching_lines_draw_add"
    bl_label = "Add this new stitching line"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        
        for obj in bpy.data.collections[COLL_NAME_FSP_SL].objects:
            if 'tmp' in obj.name:
                new_name = "new_stitching_line_" + str(len(bpy.data.collections[COLL_NAME_FSP_SL].objects)-1)
                bpy.data.grease_pencils[obj.name].name = new_name
                obj.name = new_name
        
        return {'FINISHED'}





class FSP_AddStitchingLines_draw_finish(Operator):
    bl_idname = "object.fsp_add_stitching_lines_draw_finish"
    bl_label = "Finish adding stitching lines"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        dt = bpy.types.Scene.solver_data
        fsp = dt.full_smocking_pattern
        
        all_sp, all_sp_lid, all_sp_pid = refresh_stitching_lines(fsp)
        
        fsp.update_stitching_lines(all_sp, all_sp_lid, all_sp_pid)
        usp_loc, fsp_loc = update_usp_fsp_location()
        # usp.plot(usp_loc)
        fsp.plot(fsp_loc)

                    
        return {'FINISHED'}



class FSP_EditMesh_start(Operator):
    bl_idname = "object.fsp_edit_mesh_start"
    bl_label = "Edit the mesh of the smocking pattern"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        
        bpy.ops.object.select_all(action='DESELECT')
        
        bpy.data.objects[MESH_NAME_FSP].select_set(True)
        
        bpy.ops.object.mode_set(mode = 'EDIT')

        select_mode = context.scene.fsp_edit_selection
        if select_mode == "V":
            bpy.context.tool_settings.mesh_select_mode = (True, False, False) 
        elif select_mode == "E":
            bpy.context.tool_settings.mesh_select_mode = (False, True, False)
        elif select_mode == "F":
            bpy.context.tool_settings.mesh_select_mode = (False, False, True) 
        
        return {'FINISHED'}
    
    


class FSP_EditMesh_done(Operator):
    bl_idname = "object.fsp_edit_mesh_done"
    bl_label = "Finish the edits and update the smocking pattern"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        print('not done yet :/')
        return {'FINISHED'}



    
class FSP_EditMesh_radial_grid(Operator):
    bl_idname = "object.fsp_deform_regular_into_radial_grid"
    bl_label = "Deform the pattern into radial grid"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        # if ratio = 1: the radial grid cell will be close to a square
        ratio = context.scene.radial_grid_ratio


        dt = bpy.types.Scene.solver_data

        fsp = dt.full_smocking_pattern


        fsp_new = deform_fsp_into_radial_grid(fsp, ratio)

        initialize_pattern_collection(COLL_NAME_FSP_TMP, COLL_NAME_FSP_TMP_SL)

        
        fsp_new.plot((-fsp_new.return_pattern_width()-1, 0, 0))

        dt.tmp_fsp = fsp_new

        return {'FINISHED'}




            



class FSP_CombinePatterns_load_first(Operator):
    bl_idname = "object.fsp_combine_patterns_load_first"
    bl_label = "Load the first pattern"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        initialize_pattern_collection(COLL_NAME_P1, COLL_NAME_P1_SL)
        initialize_pattern_collection(COLL_NAME_FSP_TMP, COLL_NAME_FSP_TMP_SL)
        

        # load the exiting smocking pattern to my_coll
        file_name = bpy.path.abspath(context.scene.file_import_p1)
        
        fsp = read_obj_to_fsp(file_name, MESH_NAME_P1, COLL_NAME_P1, COLL_NAME_P1_SL, "Pattern 01")

        dt = bpy.types.Scene.solver_data
        dt.tmp_fsp1 = fsp # save the data to scene

        dt.tmp_fsp = []

        fsp1_loc, fsp2_loc, _ = update_tmp_pattern_location()
        
        if dt.tmp_fsp2 != []:
            dt.tmp_fsp2.plot(fsp2_loc)

        dt.tmp_fsp1.plot(fsp1_loc)

        return {'FINISHED'}



    
    
class FSP_CombinePatterns_load_second(Operator):
    bl_idname = "object.fsp_combine_patterns_load_second"
    bl_label = "Load the second pattern"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):     
        initialize_pattern_collection(COLL_NAME_P2, COLL_NAME_P2_SL)
        initialize_pattern_collection(COLL_NAME_FSP_TMP, COLL_NAME_FSP_TMP_SL)

    
        # load the exiting smocking pattern to my_coll
        file_name = bpy.path.abspath(context.scene.file_import_p2)
        
        fsp = read_obj_to_fsp(file_name, MESH_NAME_P2, COLL_NAME_P2, COLL_NAME_P2_SL, "Pattern 02")

        dt = bpy.types.Scene.solver_data
        dt.tmp_fsp2 = fsp # save the data to scene

        dt.tmp_fsp = []

        fsp1_loc, fsp2_loc, _ = update_tmp_pattern_location()

        if dt.tmp_fsp1 != []:
            dt.tmp_fsp1.plot(fsp1_loc)

        dt.tmp_fsp2.plot(fsp2_loc)

        return {'FINISHED'}


class FSP_CombinePatterns_assign_to_first(Operator):
    bl_idname = "object.fsp_combine_patterns_assign_to_first"
    bl_label = "Combined two patterns"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        initialize_pattern_collection(COLL_NAME_P1, COLL_NAME_P1_SL)
        initialize_pattern_collection(COLL_NAME_FSP_TMP, COLL_NAME_FSP_TMP_SL)


        dt = bpy.types.Scene.solver_data
        fsp = dt.full_smocking_pattern

        dt.tmp_fsp = []

        if fsp == []:
            print('ERROR: there is no full smocking pattern in the scene')
        else:
            dt.tmp_fsp1 = SmockingPattern(fsp.V, fsp.F, fsp.E,
                 fsp.stitching_points,
                 fsp.stitching_points_line_id,
                 fsp.stitching_points_patch_id,
                 fsp.stitching_points_vtx_id,
                 MESH_NAME_P1, 
                 COLL_NAME_P1,
                 COLL_NAME_P1_SL,
                 "Pattern 01")
            fsp1_loc, fsp2_loc, _ = update_tmp_pattern_location()

            if dt.tmp_fsp2 != []:
                dt.tmp_fsp2.plot(fsp2_loc)

            dt.tmp_fsp1.plot(fsp1_loc)

        return {'FINISHED'}


class FSP_CombinePatterns_assign_to_second(Operator):
    bl_idname = "object.fsp_combine_patterns_assign_to_second"
    bl_label = "Combined two patterns"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):        
        initialize_pattern_collection(COLL_NAME_P2, COLL_NAME_P2_SL)
        initialize_pattern_collection(COLL_NAME_FSP_TMP, COLL_NAME_FSP_TMP_SL)


        dt = bpy.types.Scene.solver_data
        fsp = dt.full_smocking_pattern

        dt.tmp_fsp = []

        if fsp == []:
            print('ERROR: there is no full smocking pattern in the scene')
        else:
            dt.tmp_fsp2 = SmockingPattern(fsp.V, fsp.F, fsp.E,
                 fsp.stitching_points,
                 fsp.stitching_points_line_id,
                 fsp.stitching_points_patch_id,
                 fsp.stitching_points_vtx_id,
                 MESH_NAME_P2, 
                 COLL_NAME_P2,
                 COLL_NAME_P2_SL,
                 "Pattern 02")

            fsp1_loc, fsp2_loc, _ = update_tmp_pattern_location()

            if dt.tmp_fsp1 != []:
                dt.tmp_fsp1.plot(fsp1_loc)
                
            dt.tmp_fsp2.plot(fsp2_loc)

        return {'FINISHED'}







class FSP_CombinePatterns(Operator):
    bl_idname = "object.fsp_combine_patterns"
    bl_label = "Combine two patterns"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        
        initialize_pattern_collection(COLL_NAME_FSP_TMP, COLL_NAME_FSP_TMP_SL)



        dt = bpy.types.Scene.solver_data
        fsp1 = dt.tmp_fsp1
        fsp2 = dt.tmp_fsp2

        if fsp1 == [] or fsp2 == []:
            print("Error: Load or create two patterns first!")
        else:
            axis = context.scene.combine_direction
            dist = context.scene.combine_space
            shift = context.scene.combine_shift
            
            fsp = combine_two_smocking_patterns(fsp1, fsp2, axis, dist, shift)
        
            # save the combined data to scene
            dt.tmp_fsp = fsp


            # udpate the patterns
            fsp1_loc, fsp2_loc, fsp3_loc = update_tmp_pattern_location()

            dt.tmp_fsp1.plot(fsp1_loc)
            dt.tmp_fsp2.plot(fsp2_loc)
            dt.tmp_fsp.plot(fsp3_loc)


        return {'FINISHED'}




class FSP_confirm_tmp_fsp(Operator):
    "Assign the temporary pattern to current full smocking pattern"
    bl_idname = "object.fsp_confirm_temp_fsp"
    bl_label = "Confirm temprary smocking pattern"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        dt = bpy.types.Scene.solver_data
        
        fsp = dt.tmp_fsp

        if fsp == []:
            print("Error: Don't have a temporary pattern!")
        else:
            fsp_new = SmockingPattern(fsp.V, fsp.F, fsp.E,
                 fsp.stitching_points,
                 fsp.stitching_points_line_id,
                 fsp.stitching_points_patch_id,
                 fsp.stitching_points_vtx_id,
                 MESH_NAME_FSP, 
                 COLL_NAME_FSP,
                 COLL_NAME_FSP_SL,
                 "Full Smocking Pattern")
            dt.full_smocking_pattern = fsp_new

            _, fsp_loc = update_usp_fsp_location()

            dt.full_smocking_pattern.plot(fsp_loc)
            dt.tmp_fsp = []

            initialize_pattern_collection(COLL_NAME_FSP_TMP, COLL_NAME_FSP_TMP_SL)
            initialize_pattern_collection(COLL_NAME_USP, COLL_NAME_USP_SL)
                     
   
        return {'FINISHED'}



class SG_draw_graph(Operator):
    bl_idname = "object.sg_draw_graph"
    bl_label = "Draw the smocked graph"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        dt = bpy.types.Scene.solver_data

        fsp = dt.full_smocking_pattern

        start_time = time.time()
        SG = SmockedGraph(fsp)
        msg = 'runtime: extract smocked graph: %f second' % (time.time() - start_time)
        bpy.types.Scene.sl_props.runtime_log.append(msg)

        dt.smocked_graph = SG # save the data to the scene

        SG.info()
        SG.plot()

        return {'FINISHED'}




class PrepareCylinderArapOperator(Operator):
  bl_idname = "object.prepare_cylinder_arap_operator"
  bl_label = "Prepare ARAP"
  bl_options = {'REGISTER', 'UNDO'}

  def execute(self, context):
    dt = bpy.types.Scene.solver_data
    g_V, g_F, g_mesh, g_iters, g_arap, g_constraints = dt.arap_data
    fsp = dt.full_smocking_pattern
    sg = dt.smocked_graph
    weight = context.scene.arap_constraint_weight
    g_arap, g_constraints = prepare_cylinder_arap(g_V.T, g_F, g_arap, fsp, sg, weight)
    dt.arap_data[4] = g_arap
    dt.arap_data[5] = g_constraints

    return {'FINISHED'}

class PrepareArapOperator(Operator):
  bl_idname = "object.prepare_arap_operator"
  bl_label = "Prepare ARAP"
  bl_options = {'REGISTER', 'UNDO'}

  def execute(self, context):
    dt = bpy.types.Scene.solver_data
    if len(dt.embeded_graph) == 0:
      bpy.ops.object.sg_embed_graph()
    fsp = dt.full_smocking_pattern
    sg = dt.smocked_graph
    weight = context.scene.arap_constraint_weight
    # global g_V, g_F, g_mesh, g_iters, g_arap, g_constraints
    g_arap, g_constraints, g_F, UV = prepare_arap(dt.embeded_graph, fsp, sg, weight)
    g_V = np.concatenate((UV, np.zeros([UV.shape[0], 1])), axis=1).T
    g_iters = 100

    # Show the result.
    if "Smocked" in bpy.data.meshes:
      bpy.data.meshes.remove(bpy.data.meshes["Smocked"])
      bpy.data.collections.remove(bpy.data.collections['smocked_collection'])
    smocked_mesh = bpy.data.meshes.new("Smocked")
    smocked_mesh.from_pydata(g_V.T,[],g_F)
    
    smocked_mesh.update()
    g_mesh = smocked_mesh
    # Set the arap data in the global structure.
    dt.arap_data = [g_V, g_F, g_mesh, g_iters, g_arap, g_constraints]

    # Add uv layer
    uvlayer = smocked_mesh.uv_layers.new()
    for face in smocked_mesh.polygons:
      for vert_idx, loop_idx in zip(face.vertices, face.loop_indices):
          uvlayer.data[loop_idx].uv = (UV[vert_idx, 0] / 20.0, UV[vert_idx, 1] / 20.0)
    
    obj = bpy.data.objects.new('smocked_obj', smocked_mesh)
    obj.location = (0, 0, 4)
    obj.data.materials.append(bpy.data.materials['Fabric035'])
    for f in obj.data.polygons:
      f.use_smooth = True
    smocked_collection = bpy.data.collections.new('smocked_collection')
    bpy.context.scene.collection.children.link(smocked_collection)
    smocked_collection.objects.link(obj)

    return {'FINISHED'}


class MagicButtonOperator(Operator):
    bl_idname = "objects.magic_button_operator"
    bl_label = "Simulate"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        bpy.ops.object.sg_embed_graph()
        bpy.ops.object.prepare_arap_operator()
        bpy.ops.arap.realtime_operator()

        return {'FINISHED'}



class SG_embed_graph(Operator):
    bl_idname = "object.sg_embed_graph"
    bl_label = "Embed the smocked graph"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        dt = bpy.types.Scene.solver_data
        fsp = dt.full_smocking_pattern
        start_time = time.time()
        dt.smocked_graph = SmockedGraph(fsp)
        sg = dt.smocked_graph
    
        C_underlay_eq, C_pleat_eq = sg.return_distance_constraint()


        #----------------- embed the underlay 
        X_underlay_ini = sg.V[sg.vid_underlay, 0:2]    
        start_time = time.time()
        res_underlay = np.array(cpp_smocking_solver.embed_underlay(X_underlay_ini, C_underlay_eq))
        msg = 'optimization: embed the underlay graph: %f second' % (time.time() - start_time)
        bpy.types.Scene.sl_props.runtime_log.append(msg)

        X = res_underlay.reshape(int(len(res_underlay)),2)
        X_underlay = np.concatenate((X, np.zeros((len(X),1))), axis=1)

        #------------------ embed the pleat
        X_pleat = sg.V[sg.vid_pleat, 0:2]

        # option 01: initialize from the original position 
        X_pleat = np.concatenate((X_pleat, np.ones((len(X_pleat),1))), axis=1)
        
        w_pleat_embed = context.scene.pleat_max_embed
        w_pleat_eq = context.scene.pleat_eq
        w_pleat_var = context.scene.pleat_variance
        start_time = time.time()
        res_pleat = np.array(cpp_smocking_solver.embed_pleats(
          X_underlay, X_pleat, C_pleat_eq, w_pleat_var, w_pleat_embed, w_pleat_eq))
        msg = 'optimization: embed the underlay graph: %f second' % (time.time() - start_time)
        bpy.types.Scene.sl_props.runtime_log.append(msg)

        X_pleat = res_pleat.reshape(int(len(res_pleat)), 3)
        X_all = np.concatenate((X_underlay, X_pleat), axis=0)

        print_runtime()

        dt.embeded_graph = X_all

        # clean_objects_in_collection(COLL_NAME_SG)
        # trans = [0,-10, 0]
        # for eid in range(sg.ne):
        #     vtxID = sg.E[eid, :]
        #     pos = X_all[vtxID, :] + trans
        #     if sg.is_edge_underlay(eid):
        #         col = col_gray
        #     else:
        #         col = col_red
        #     draw_stitching_line(pos, col, "embed_" + str(eid), int(STROKE_SIZE/2), COLL_NAME_SG)
        
        return {'FINISHED'}




# ========================================================================
#                          Draw the Panel
# ========================================================================

class debug_panel(bpy.types.Panel):
    bl_label = "Debug :/"
    bl_idname = "SD_PT_debug"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SmockingDesign"
    bl_options ={"DEFAULT_CLOSED"}
    
    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False  # No animation.

        c = layout.column()
        row = c.row()
        row.operator(debug_clear.bl_idname, text="clear everything", icon='QUIT')
        row = self.layout.row()
        row.operator(debug_print.bl_idname, text="print data in scene", icon='GHOST_ENABLED')
        row = self.layout.row()
        row.operator(debug_func.bl_idname, text="test function", icon="GHOST_DISABLED")
        row = self.layout.row()
        row.operator(RealtimeArapOperator.bl_idname, text="arap", icon="MOD_MESHDEFORM")
        row = self.layout.row()
        row.operator(CreateClothSimOperator.bl_idname, text="Cloth sim", icon="MOD_MESHDEFORM")
        row = self.layout.row()
        row.operator(DirectArapOperator.bl_idname, text="Direct smock ARAP", icon="MOD_MESHDEFORM")
        row = self.layout.row()
        row.operator(debug_render.bl_idname, text="Render", icon='RENDER_STILL')
        row = self.layout.row()
        row.operator(isometric_distortion.bl_idname, text="Show isometryic distortion", icon='RENDER_STILL')
        row = self.layout.row()
        row.prop(context.scene, 'if_highlight_button', toggle=False)



class UnitGrid_panel:
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SmockingDesign"
    bl_options ={"HEADER_LAYOUT_EXPAND"}
     
    

class UNITGRID_PT_main(UnitGrid_panel, bpy.types.Panel):
    bl_label = "Unit Smocking Pattern"
    bl_idname = "SD_PT_unit_grid_main"
    bl_options ={"DEFAULT_CLOSED"}
            
    def draw(self, context):
        pass

    
class UNITGRID_PT_create(UnitGrid_panel, bpy.types.Panel):
    bl_parent_id = 'SD_PT_unit_grid_main'
    bl_label = "Create a New Pattern"
    bl_options ={"DEFAULT_CLOSED"}
    
    
    def draw(self, context):
        props = context.scene.sl_props
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False  # No animation.

        
        layout.label(text= "Generate A Grid for Drawing:")
        box = layout.box()
        box.row()
        box.row().prop(context.scene,'base_x')
        box.row().prop(context.scene,'base_y')

        row = box.row()
        row.alert=context.scene.if_highlight_button
        row.operator(USP_CreateGrid.bl_idname, text="Generate Grid", icon="GRID")
        box.row()
        
        
        layout.label(text= "Draw A Stitching Line")
        box = layout.box()
        box.row()
        row = box.row()
        if(not context.window_manager.usp_drawing_started):
            row.alert=context.scene.if_highlight_button
            row.operator(USP_SelectStitchingPoint.bl_idname, text="Draw", icon='GREASEPENCIL')
        else:
            row.alert=context.scene.if_highlight_button
            row.operator(USP_FinishCurrentDrawing.bl_idname, text="Done", icon='CHECKMARK')
       

        row.alert=context.scene.if_highlight_button
        row.operator(USP_SaveCurrentStitchingLine.bl_idname, text="Add", icon='ADD')
        
        box.row()
        row = box.row()
        row.alert=context.scene.if_highlight_button
        row.operator(USP_FinishPattern.bl_idname, text="Finish Unit Pattern Design", icon='HEART')
        box.row()
        
         
        layout.label(text= "Export Current Unit Smocking Pattern")
        box = layout.box()
        box.row()  
        box.row().prop(context.scene, 'path_export')
        box.row().prop(context.scene, 'filename_export')
        row = box.row() 
        row.alert=context.scene.if_highlight_button
        row.operator(ExportUnitPattern.bl_idname, text="Export", icon='EXPORT')
        box.row()
        
                 




class UNITGRID_PT_load(UnitGrid_panel, bpy.types.Panel):
    bl_parent_id = 'SD_PT_unit_grid_main'
    bl_label = "Load Existing Pattern"
    bl_options ={"DEFAULT_CLOSED"}
    
    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False  # No animation.

#        layout.label(text= "Load an Existing Pattern")
        box = layout.box()
        box.row()
        box.row().prop(context.scene, 'path_import')
        row = box.row()
        row.alert=context.scene.if_highlight_button
        row.operator(ImportUnitPattern.bl_idname, text="Import", icon='IMPORT')
        box.row()




class FullGrid_panel():
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SmockingDesign"
    bl_options ={"HEADER_LAYOUT_EXPAND"}
     
    



class FULLGRID_PT_main(FullGrid_panel, bpy.types.Panel):
    bl_label = "Full Smocking Pattern"
    bl_idname = "SD_PT_full_grid_main"
    bl_options ={"DEFAULT_CLOSED"}
            
    def draw(self, context):
        pass



        
class FULLGRID_PT_tile(FullGrid_panel, bpy.types.Panel):
    bl_label = "Tile Unit Grid"
    bl_parent_id = 'SD_PT_full_grid_main'
    bl_options ={"DEFAULT_CLOSED"}

    
    def draw(self, context):
        
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False  # No animation.

        box = layout.box()
        box.row()
        box.row().prop(context.scene,'num_x')
        box.row().prop(context.scene,'num_y')
    
        box.row().prop(context.scene,'shift_x')
        box.row().prop(context.scene,'shift_y')
        
        # layout.label(text= "Tiling Type:")
        # row = layout.row()
        # row.prop(context.scene, 'type_tile', expand=True)
        
        row = box.row()
        row.alert=context.scene.if_highlight_button
        row.operator(FSP_Tile.bl_idname, text="Generate by Tiling", icon='FILE_VOLUME')
        box.row()

class CreateHextGridPanel(FullGrid_panel, bpy.types.Panel):
    bl_label = "Create honeybomb grid"
    bl_parent_id = 'SD_PT_full_grid_main'
    bl_options ={"DEFAULT_CLOSED"}

    
    def draw(self, context):
        
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False  # No animation.

        box = layout.box()
        box.row()
        box.row().prop(context.scene,'num_x')
        box.row().prop(context.scene,'num_y')

        row = box.row()
        row.alert=context.scene.if_highlight_button
        row.operator(HoneycombGrid.bl_idname, text="Generate", icon='FILE_VOLUME')


class embedding_panel(bpy.types.Panel):
  bl_label = "Emedding"
  bl_idname = "embedding_panel"
  bl_space_type = "VIEW_3D"
  bl_region_type = "UI"
  bl_category = "SmockingDesign"
  bl_options = {"DEFAULT_CLOSED"}
  
  def draw(self, context):
    layout = self.layout
    layout.use_property_split = True

    layout.label(text="Embedding weights")
    box = layout.box()
    box.row().prop(context.scene, 'pleat_eq')
    box.row().prop(context.scene, 'pleat_max_embed')
    box.row().prop(context.scene, 'pleat_variance')

    layout.row().operator(SG_embed_graph.bl_idname, text="Embed", icon='IMPORT')

class arap_panel(bpy.types.Panel):
  bl_label = "ARAP"
  bl_idname = "arap_panel"
  bl_space_type = "VIEW_3D"
  bl_region_type = "UI"
  bl_category = "SmockingDesign"
  bl_options = {"DEFAULT_CLOSED"}
  
  def draw(self, context):
    pass

class ArapGridPanel(bpy.types.Panel):
  bl_label = "Pattern ARAP"
  bl_parent_id = 'arap_panel'
  bl_space_type = "VIEW_3D"
  bl_region_type = "UI"
  bl_category = "SmockingDesign"
  bl_options = {"DEFAULT_CLOSED"}
  
  def draw(self, context):
    layout = self.layout
    layout.use_property_split = True

    layout.label(text="Arap parameters")
    box = layout.box()
    box.row().prop(context.scene, 'arap_constraint_weight')
    box.row().prop(context.scene, 'arap_num_iters')

    layout.row().operator(PrepareArapOperator.bl_idname, text="Prepare", icon='IMPORT')
    layout.row().operator(RealtimeArapOperator.bl_idname, text="Run (esc to stop)", icon='IMPORT')

class CylinderArap(bpy.types.Panel):
  bl_label = "Parameterization ARAP"
  bl_parent_id = 'arap_panel'
  bl_space_type = "VIEW_3D"
  bl_region_type = "UI"
  bl_category = "SmockingDesign"
  bl_options = {"DEFAULT_CLOSED"}
  
  def draw(self, context):
    layout = self.layout
    layout.use_property_split = True

    layout.label(text="Arap parameters")
    box = layout.box()
    box.row().prop(context.scene, 'arap_constraint_weight')
    box.row().prop(context.scene, 'arap_num_iters')

    layout.row().operator(PrepareCylinderArapOperator.bl_idname, text="Prepare", icon='IMPORT')
    layout.row().operator(RealtimeArapOperator.bl_idname, text="Run (esc to stop)", icon='IMPORT')
      


class FULLGRID_PT_combine_patterns(FullGrid_panel, bpy.types.Panel):
    bl_label = "Combine Two Patterns"
    bl_parent_id = 'SD_PT_full_grid_main'
    bl_options ={"DEFAULT_CLOSED"}
    
    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False  # No animation.

        layout.label(text= "Get Two Smocking Patterns")
        row = layout.row()

        box = row.box()
        box.label(text="Load from Saved Files")

        col = box.column(align=True)
        split = col.split(factor=0.7)
        row = split.row()
        row.prop(context.scene, 'file_import_p1')
        row = split.split() 
        row.alert=context.scene.if_highlight_button
        row.operator(FSP_CombinePatterns_load_first.bl_idname, text="Import P1", icon='IMPORT')
        
        col = box.column(align=True)
        split = col.split(factor=0.7)
        row = split.row()
        row.prop(context.scene, 'file_import_p2')      
        row = split.split()
        row.alert=context.scene.if_highlight_button
        row.operator(FSP_CombinePatterns_load_second.bl_idname, text="Import P2", icon='IMPORT')

        box.label(text= "Assign Current Pattern")
        row = box.row()
        row.alert=context.scene.if_highlight_button
        row.operator(FSP_CombinePatterns_assign_to_first.bl_idname, text="assign to P1", icon='FORWARD')

        row.alert=context.scene.if_highlight_button
        row.operator(FSP_CombinePatterns_assign_to_second.bl_idname, text="assign to P2", icon='FORWARD')
        box.row()

        

        layout.label(text= "Parameters")
        row = layout.row()
        box = row.box()
        box.row()
        box.row().prop(context.scene, 'combine_direction', expand=True)
        box.prop(context.scene, 'combine_space')
        box.prop(context.scene, 'combine_shift')
        box.row()
        


        
        row = layout.row()
        row = layout.row()
        row.alert=context.scene.if_highlight_button
        row.operator(FSP_CombinePatterns.bl_idname, text="Combined P1 and P2", icon="NODE_COMPOSITING")

        row = layout.row()
        row = layout.row()
        row.alert=context.scene.if_highlight_button
        row.operator(FSP_confirm_tmp_fsp.bl_idname, text="Assign to Full Smocking Pattern", icon="FORWARD")







class FULLGRID_PT_edit_pattern(FullGrid_panel, bpy.types.Panel):
    bl_label = "Edit Current Pattern"
    bl_parent_id = 'SD_PT_full_grid_main'
    bl_options ={"DEFAULT_CLOSED"}
    
    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False  # No animation.


        layout.label(text= "Delete Stitching Lines")
        box = layout.box()
        box.row()
        row = box.row()
        row.alert=context.scene.if_highlight_button
        row.operator(FSP_DeleteStitchingLines_start.bl_idname, text="Delete", icon="PANEL_CLOSE")
        
        row.alert=context.scene.if_highlight_button
        row.operator(FSP_DeleteStitchingLines_done.bl_idname, text="Done", icon="CHECKMARK")
        box.row()
        
        layout.label(text= "Add New Stitching Lines")
        box = layout.box()
        box.row()
        row = box.row()

        if(not context.window_manager.fsp_drawing_started):
            row.alert=context.scene.if_highlight_button
            row.operator(FSP_AddStitchingLines_draw_start.bl_idname, text="Draw", icon='GREASEPENCIL')
        else:
            row.alert=context.scene.if_highlight_button
            row.operator(FSP_AddStitchingLines_draw_end.bl_idname, text="Done", icon='CHECKMARK')
        
        
        row.alert=context.scene.if_highlight_button
        row.operator(FSP_AddStitchingLines_draw_add.bl_idname, text="Add", icon='ADD')
        

        row = box.row()
        row.alert=context.scene.if_highlight_button
        row.operator(FSP_AddStitchingLines_draw_finish.bl_idname, text="Finish Adding Extra Stitching Lines", icon='HEART')
        box.row()
        
        
        layout.label(text= "Edit the Smocking Grid")
        box = layout.box()
        box.row()
        box.row().prop(context.scene, 'fsp_edit_selection', expand=True)

        row = box.row()
        row.alert=context.scene.if_highlight_button
        row.operator(FSP_EditMesh_start.bl_idname, text="Edit", icon="EDITMODE_HLT")


        row.alert=context.scene.if_highlight_button
        row.operator(FSP_EditMesh_done.bl_idname, text="Done", icon="CHECKMARK")
        box.row()


        layout.label(text="Deform into Radial Grid")
        box = layout.box()
        box.row()
        
        box.row().prop(context.scene, 'radial_grid_ratio')
        row = box.row()
        row.alert=context.scene.if_highlight_button
        row.operator(FSP_EditMesh_radial_grid.bl_idname, text="Deform", icon="MOD_SIMPLEDEFORM")

        row = box.row()
        row.alert=context.scene.if_highlight_button
        row.operator(FSP_confirm_tmp_fsp.bl_idname, text="Assign to Full Smocking Pattern", icon="FORWARD")
        box.row()
        
        
        

class FULLGRID_PT_add_margin(FullGrid_panel, bpy.types.Panel):
    bl_label = "Add Margin to Current Pattern"
    bl_parent_id = 'SD_PT_full_grid_main'
    bl_options ={"DEFAULT_CLOSED"}

    
    def draw(self, context):
        
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False  # No animation.

        box = layout.box()
        box.row()
        box.row().prop(context.scene,'margin_top')
        box.row().prop(context.scene,'margin_bottom')
        box.row().prop(context.scene,'margin_left')
        box.row().prop(context.scene,'margin_right')
        
        
        row = box.row()
        row.alert=context.scene.if_highlight_button
        row.operator(FSP_AddMargin.bl_idname, text="Add Margin to Pattern (Preview)", icon='OBJECT_DATAMODE')

        row = box.row()
        row.alert=context.scene.if_highlight_button
        row.operator(FSP_confirm_tmp_fsp.bl_idname, text="Assign to Full Smocking Pattern", icon="FORWARD")
        box.row()
        


class FULLGRID_PT_export_mesh(FullGrid_panel, bpy.types.Panel):
    bl_label = "Export Current Pattern to Mesh"
    bl_parent_id = 'SD_PT_full_grid_main'
    bl_options ={"DEFAULT_CLOSED"}
    
    def draw(self, context):
        # TODO: the alignment is off :/ dunno how to fix it
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False  # No animation.


        box = layout.box()
        box.row()
        
        box.row().prop(context.scene, 'path_export_fullpattern')
        box.row().prop(context.scene, 'filename_export_fullpattern')
        box.row().prop(context.scene, 'export_format', expand=True)
        
        row = box.row()
        row.alert=context.scene.if_highlight_button
        row.operator(FSP_Export.bl_idname, text="Export", icon='EXPORT')
        box.row()
        
class FULLGRID_PT_import_mesh(FullGrid_panel, bpy.types.Panel):
    bl_label = "Import full smocking pattern"
    bl_parent_id = 'SD_PT_full_grid_main'
    bl_options ={"DEFAULT_CLOSED"}
    
    def draw(self, context):
      layout = self.layout
      layout.use_property_split = True
      layout.use_property_decorate = False  # No animation.

      box = layout.box()
      box.row()
      
      box.row().prop(context.scene, 'path_import_fullpattern')
      
      row = box.row()
      row.alert=context.scene.if_highlight_button
      row.operator(FSP_Import.bl_idname, text="Import", icon='IMPORT')
      box.row()



class FastSmock_panel(bpy.types.Panel):
    bl_label = "Fast Smocking"
    bl_idname = "SD_PT_fast_smock"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SmockingDesign"
    bl_options ={"DEFAULT_CLOSED"}

    def draw(self, context):

        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False  # No animation.

        layout.label(text= "Load an Existing Pattern")
        box = layout.box()
        box.row()
        box.row().prop(context.scene, 'path_import')
        row = box.row()
        row.alert=context.scene.if_highlight_button
        row.operator(ImportUnitPattern.bl_idname, text="Import", icon='IMPORT')
        box.row()


        layout.label(text= "Tile the Unit Pattern")
        box = layout.box()
        box.row()
        box.row().prop(context.scene,'num_x')
        box.row().prop(context.scene,'num_y')
        row = box.row()
        row.alert=context.scene.if_highlight_button
        row.operator(FSP_Tile.bl_idname, text="Generate by Tiling", icon='FILE_VOLUME')
        box.row()

        row = layout.row()
        row.alert=context.scene.if_highlight_button
        row.operator(MagicButtonOperator.bl_idname, text="Simulate the Smocking Pattern", icon='FILE_VOLUME')
        layout.row()



        
class SmockedGraph_panel(bpy.types.Panel):
    bl_label = "Smocked Graph"
    bl_idname = "SD_PT_smocked_graph"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SmockingDesign"
    bl_options ={"DEFAULT_CLOSED"}
   
    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False  # No animation.

        layout.label(text="Show the Smocked Graph")
        box = layout.row().box()
        box.row()
        box.row().prop(context.scene, "graph_select_type", expand=True)
        box.row().prop(context.scene, "graph_highlight_type", expand=True)
        row = box.row()
        row.alert=context.scene.if_highlight_button
        row.operator(SG_draw_graph.bl_idname, text='Visualize the Smocked Graph', icon="GREASEPENCIL")
        box.row()
        
        
        layout.label(text="Embed the Smocked Graph")
        box = layout.row().box()
        box.label(text="Optimization Parameters")
        box.row().prop(context.scene, "opti_init_type", expand=True)
        box.row().prop(context.scene, "opti_init_pleat_height")
        box.row().prop(context.scene, "opti_dist_preserve", expand=True)
        box.row().prop(context.scene, "opti_if_add_delaunay", expand=True)
        box.row().prop(context.scene, "opti_mtd")
        box.row().prop(context.scene, "opti_tol", slider=False)
        box.row().prop(context.scene, "opti_maxIter", slider=False)
        box.row().prop(context.scene, "opti_if_display")
        




        row = box.row()
        row.alert=context.scene.if_highlight_button
        row.operator(SG_embed_graph.bl_idname, text='Embed the Smocked Graph', icon="GREASEPENCIL")
        box.row()
        
        
        
# ========================================================================
#                          Registration
# ========================================================================

wm = bpy.types.WindowManager
wm.usp_drawing_started = bpy.props.BoolProperty(default=False)
wm.fsp_drawing_started = bpy.props.BoolProperty(default=False)


_classes = [
    
    debug_clear,
    debug_print,
    debug_render,
    isometric_distortion,
    debug_func,

    # UI panels.

    # Debug.
    debug_panel,
    # Unit grid panel.
    UNITGRID_PT_main,
    UNITGRID_PT_create,
    UNITGRID_PT_load,
    # Full smocking pattern panel.
    FULLGRID_PT_main,
    FULLGRID_PT_tile,
    CreateHextGridPanel,
    FULLGRID_PT_combine_patterns,
    FULLGRID_PT_edit_pattern,
    FULLGRID_PT_add_margin,
    FULLGRID_PT_export_mesh,
    FULLGRID_PT_import_mesh,
    # Embedding panel
    embedding_panel,
    # Arap panel
    arap_panel,
    ArapGridPanel,
    CylinderArap,
    # Smocked graph panel.
    SmockedGraph_panel,
    FastSmock_panel,
    
    USP_CreateGrid,
    USP_SaveCurrentStitchingLine,
    USP_SelectStitchingPoint,
    USP_FinishCurrentDrawing,
    USP_FinishPattern,
    ExportUnitPattern,
    ImportUnitPattern,
        
    HoneycombGrid,
    FSP_Tile,
    FSP_AddMargin,
    FSP_Export,
    FSP_Import,
    FSP_DeleteStitchingLines_start,
    FSP_DeleteStitchingLines_done,
    
    FSP_EditMesh_start,
    FSP_EditMesh_done,
    FSP_EditMesh_radial_grid,

    FSP_AddStitchingLines_draw_start,
    FSP_AddStitchingLines_draw_end,
    FSP_AddStitchingLines_draw_add,    
    FSP_AddStitchingLines_draw_finish,
    FSP_CombinePatterns_load_first,
    FSP_CombinePatterns_load_second,
    FSP_CombinePatterns_assign_to_first,
    FSP_CombinePatterns_assign_to_second,
    FSP_CombinePatterns,
    FSP_confirm_tmp_fsp,
    
    SG_draw_graph,
    SG_embed_graph,
    
    MagicButtonOperator,

    

    PrepareArapOperator,
    PrepareCylinderArapOperator,
    RealtimeArapOperator,
    CreateClothSimOperator,
    DirectArapOperator,
 ]


def register():
    for (prop_name, prop_value) in PROPS:
        setattr(bpy.types.Scene, prop_name, prop_value)
        
    for cls in _classes:
        bpy.utils.register_class(cls)
        
    bpy.types.Scene.sl_props = GlobalVariable()
    bpy.types.Scene.solver_data = SolverData()
    

def unregister():
    for cls in _classes:
        bpy.utils.unregister_class(cls)
        
    del bpy.types.Scene.sl_props    
    del bpy.tpyes.Scene.solver_data

if __name__ == "__main__":
    register()
#    bpy.ops.object.export_unit_pattern('INVOKE_DEFAULT')




# ========================================================================
#                         as useless as jing
# ========================================================================
# build-in subdivision
#        scale = max(base_x, base_y)
#        # subdivison a bit buggy here
#        bpy.ops.mesh.primitive_grid_add(size=scale, 
#                                        location=(base_x/2, base_y/2,0),
#                                        x_subdivisions=base_x + 1,
#                                        y_subdivisions=base_y + 1) 






# TODO: some bug here - cannot click start twice :/
#class startAddStitchingLines(Operator):
#    """initialization for adding stitching lines"""
#    bl_idname = "edit.initialize_add_stitching_lines"
#    bl_label = "Initialization for Adding Stitching Lines"
#    bl_options = {'REGISTER', 'UNDO'}
#    
#    def execute(self, context):
#        
#        props = bpy.context.scene.sl_props
#        
#        currentDrawing = [] # clear the previously selected stitching lines
#        
#        mesh = bpy.data.objects[MESH_NAME_USP]
#        select_one_object(mesh)

#        bpy.ops.object.mode_set(mode = 'EDIT') 
#        
#        return {'FINISHED'}



#class deleteCurrentStitchingLine(Operator):
#    bl_idname = "edit.delete_current_stitching_line"
#    bl_label = "delete the current one stiching line"
#    bl_options = {'REGISTER', 'UNDO'}
#    
#    def execute(self, context):
#        props = bpy.context.scene.sl_props
#        props.currentDrawing = []
#        
#        delete_all_gpencil_objects()
#        
#        draw_saved_stitching_lines(context)
#        
#        props.if_curr_drawing_is_shown = True
#        
#        return {'FINISHED'}


                          
#        mesh = generate_grid(base_x, base_y)
#        for vid in range(len(mesh.data.vertices)):
#            print(get_curr_vtx_pos(mesh, vid))




#def write_usp_old(file_name, base_x, base_y, savedStitchingLines, mesh):
#    file = open(file_name, 'w')
#    file.write('gridX\t' + str(base_x) + "\n")
#    file.write('gridY\t' + str(base_y) + "\n")
#    
#    # write the stitching lint point IDs - for blender
#    for line in savedStitchingLines:
#        file.write('sl\t')
#        for vid in line:
#            file.write(str(vid)+"\t")
#        file.write("\n")
#    

#    lid = 0
#    # write the stiching lines
#    for line in savedStitchingLines:
#        
#        for vid in line:
#            pos = get_curr_vtx_pos(mesh, vid)
#            
#            file.write('sp \t' + str(lid) + '\t')
#            
#            for val in pos:
#                file.write(str(val)+"\t")
#            file.write("\n")
#        
#        lid += 1
#        
#            
#    file.close()



# old code for creating grid

#if (isinstance(num_x, int) or isinstance(num_x, float)) and (isinstance(num_y, int) or isinstance(num_y, float)):
#        min_x = 0
#        max_x = num_x
#        
#        min_y = 0
#        max_y = num_y
#    elif isinstance(num_x, list) and isinstance(num_y, list):
#        if len(num_x) == 2 and len(num_y) == 2:
#            min_x = num_x[0]
#            max_x = num_x[1]
#            
#            min_y = num_y[0]
#            max_y = num_y[1]
#        else:
#            error('not suppored input for now:/')        
#            
#    else:
#        error('not suppored input for now:/')        
#    
#    # both min/max_x/y can be float, need to be careful
#    # we assume the stitching points have the integer coordinates
#    # the float part comes from stupid shift and margin :/
#        
#    range_x = range(int(np.ceil(min_x)), int(np.floor(max_x))+1)
#    range_y = range(int(np.ceil(min_y)), int(np.floor(max_y))+1)    
#    
#    xx = [-m_left + min_x, min_x] + list(range_x) + [max_x, max_x + m_right]
#    yy = [-m_bottom + min_y, min_y] + list(range_y) + [max_y, max_y + m_top]





# def delete_one_collection(coll_name):
#     for coll in bpy.data.collections:
#         if coll_name in coll.name:
#             for child in coll.children:
#                 bpy.data.collections.remove(child)
#                 clean_objects_in_collection(coll.name)



# def delete_all():
#     bpy.ops.object.select_all(action='SELECT')
#     bpy.ops.object.delete()


    # if fsp1 != []:
    #     fsp1_loc = [min(fsp1.V[:, 0]), min(fsp1.V[:, 1])]
    #     fsp1_loc[0] -= fsp1.return_pattern_width()
    # else:
    #     fsp1_loc = [0, 0]
    # if fsp2 != []:
    #     fsp2_loc = [min(fsp2.V[:,0]), min(fsp2.V[:, 1])]
    #     fsp1_loc[0] -= fsp2.return_pattern_width()
    #     fsp2_loc[0] -= fsp2.return_pattern_width()
    # else:
    #     fsp2_loc = fsp1_loc
    # if fsp3 != []:
    #     fsp3_loc = [min(fsp3.V[:, 0]), min(fsp3.V[:, 1])]
    #     fsp3_loc[0] -= fsp3.return_pattern_width()
    #     fsp1_loc[1] += fsp3.return_pattern_height() + LAYOUT_Y_SHIFT
    #     fsp2_loc[1] += fsp3.return_pattern_height() + LAYOUT_Y_SHIFT
    # else:
    #     fsp3_loc = [0,0]


    # if fsp1 != []:
    #     fsp1_loc[0] = -fsp1.return_pattern_width() - LAYOUT_X_SHIFT 
    #     fsp1_loc[1] = fsp1.return_pattern_height() + LAYOUT_Y_SHIFT 

    # if fsp2 != []:
    #     fsp2_loc[0] = -fsp2.return_pattern_width() - LAYOUT_X_SHIFT 
    #     fsp2_loc[1] = 0

    # if fsp3 != []: # the combined pattern
    #     fsp3_loc[0] = -fsp3.return_pattern_width() - LAYOUT_X_SHIFT
    #     fsp3_loc[1] = - fsp2.return_pattern_height() - LAYOUT_Y_SHIFT


    # a = min(fsp1_loc[0], fsp2_loc[0], fsp3_loc[0])

    # fsp1_loc[0], fsp2_loc[0], fsp3_loc[0] = a, a, a

    # if fsp3 != []: # such that fsp1 and fsp2 are above fsp3
    #     a = max(fsp3.return_pattern_height() + LAYOUT_Y_SHIFT, fsp1_loc[1], fsp2_loc[1])
    #     fsp1_loc[1], fsp2_loc[1] = a, a

    # if fsp2 != []:
    #     a = min(-fsp2.return_pattern_width() - LAYOUT_X_SHIFT, fsp1_loc[0], fsp3_loc[0])
    #     fsp1_loc[0], fsp3_loc[0] = a, a


# ========================================================================
#                         to delete - temporary backup
# ========================================================================


        # # construct smocked graph
        # fsp_vid_underlay = fsp.stitching_points_vtx_id
        # fsp_vid_pleat = np.setdiff1d(range(fsp.nv), fsp_vid_underlay)

        # # use dictionary to record the vertex correspondences between the 
        # # smocked graph (sg) and the smocking pattern (sp)
        # dict_sp2sg = {}
        # dict_sg2sp = {}

        # nl = fsp.num_stitching_lines()
        
        # # each stiching line in SP becomes a single node in SG
        # for lid in range(nl):
        #     vtxID = fsp.get_vid_in_stitching_line(lid)
        #     dict_sg2sp[lid] = vtxID

        #     for vid in vtxID:
        #         dict_sp2sg[vid] = lid

        # count = nl
        # for vid in fsp_vid_pleat:
        #     dict_sp2sg[vid] = count
        #     dict_sg2sp[count] = [vid]
        #     count += 1
        
        # # by construction we know
        # # the first nl vertices are underlay vertices
        # vid_underlay = np.array(range(nl))
        # # the rest are pleat vertices
        # vid_pleat = np.array(range(nl, len(dict_sg2sp)))


        # # vtx pos of the smocked graph
        
        # V = fsp.V

        # V_sg = np.zeros((len(dict_sg2sp),3))

        # for vid in range(len(dict_sg2sp)):
        #     vtx = dict_sg2sp[vid]
        #     if len(vtx) > 1:
        #         coord = np.mean(V[vtx, :], axis=0)
        #     else:
        #         coord = V[vtx, :]
        #     V_sg[vid, 0:2] = coord    
            

        # # find the edges
        # E = fsp.E
        # E_sg = []
        # for e in E:
        #     E_sg.append([dict_sp2sg[e[0]], dict_sp2sg[e[1]]])

        # E_sg = sort_edge(E_sg)
        
        # # category the edges
        # tmp = np.array(E_sg[:,0] < nl).astype(int) + np.array(E_sg[:, 1] < nl).astype(int)
        # print(tmp)
        
        # eid_underlay = np.array(find_index_in_list(tmp, 2))
        # eid_pleat = np.setdiff1d(range(len(E_sg)), eid_underlay)
        # print(eid_underlay)
        # print(eid_pleat)
        # # print(E_sg[eid_underlay,:])


#--------------------------------------------------------
# old formulation with equality and inequality constraints

#         #------------------

#         # find the pairwise distance constraints for the underlay graph
        
#         D = sg.return_pairwise_distance_constraint_for_underlay()

#         constrained_vtx_pair = SG_find_valid_underlay_constraints_exact(sg, D)
        
#         # embed the underlay graph

#         # find the vtx_pair where the max_dist is reached
#         E_eq = sg.E[sg.eid_underlay, :]

#         # find the vtx_pair where the max_dist forms inequality constraints
#         idx, _ = setdiffnd(constrained_vtx_pair, E_eq)
#         E_neq = constrained_vtx_pair[idx, :]

#         # formulate the constraints 
#         C_eq = []
#         for i, j in zip(E_eq[:, 0], E_eq[:, 1]):
#             C_eq.append([i, j, D[i,j]])

#         C_eq = np.array(C_eq)

#         C_neq = []
#         for i, j in zip(E_neq[:, 0], E_neq[:, 1]):
#             C_neq.append([i, j, D[i,j]])
#         C_neq = np.array(C_neq)
        

#         w_eq = 1e6
#         w_neq = 1e2
#         eps_neq = 0


#         #----------------- optimize for the underlay graph
#         X_underlay = sg.V[sg.vid_underlay, 0:2]

#         # use one equal constraint to rescale the initial embedding
#         # maybe converge faster
#         id_eq = 1
#         v1 = X_underlay[int(C_eq[id_eq, 0]), :]
#         v2 = X_underlay[int(C_eq[id_eq, 1]), :]
#         d12 = np.linalg.norm(v1 - v2)
#         e12 = C_eq[id_eq, 2]
#         scale = e12/d12

#         x0 = X_underlay.flatten()*scale


#         bounds = Bounds( -2*np.ones((len(x0),1)), 10*np.ones((len(x0),1)) )
        
#         y_ini = opti_energy_sg_underlay(x0, C_eq, C_neq, w_eq, w_neq, eps_neq)
#         start_time = time.time()
#         #--------------------- Test 01
#         for w_eq in [1e8, 1e6, 1e4, 1e2]:
#             res = minimize(opti_energy_sg_underlay, 
#                            x0, 
#                            method='Nelder-Mead',
#                            args=(C_eq, C_neq, w_eq, w_neq, eps_neq), 
#                            options=opti_get_NelderMead_solver_options())
#             x0 = res.x
            
#         msg = 'optimization: embed the underlay graph: %f second' % (time.time() - start_time)
#         bpy.types.Scene.sl_props.runtime_log.append(msg)


#         print_runtime()
        
#         y_res = opti_energy_sg_underlay(res.x, C_eq, C_neq, w_eq, w_neq, eps_neq)


#         X = res.x.reshape(int(len(res.x)/2),2)

#         X = np.concatenate((X, np.zeros((len(X),1))), axis=1)

#         X_underlay = X

#         print(y_ini, y_res)
        
#         fval_max, fval_eq, fval_neq = opti_energy_sg_underlay(res.x, C_eq, C_neq, w_eq, w_neq, eps_neq, True)
        
#         print(fval_max, fval_eq, fval_neq)


#         #
#         # print(X_underlay)
#         # print(X)
        
#         #----------------- optimize for the pleat graph
#         # constraints for the pleat graph
#         C_pleat_neq = []
#         for eid in sg.eid_pleat:
#             d = sg.return_max_dist_constraint_for_edge(eid)
#             C_pleat_neq.append([sg.E[eid, 0], sg.E[eid, 1], d])

#         C_pleat_neq = np.array(C_pleat_neq)

#         X_pleat = sg.V[sg.vid_pleat, 0:2]

#         # option 01: initialize from the original position 
#         X_pleat = np.concatenate((X_pleat, np.ones((len(X_pleat),1))), axis=1)

#         '''
#         # option 02: update the positions w.r.t. the underlay graph
#         vid = 1 # vid in X_pleat
#         vid_sg = vid + sg.nv_underlay # the vid in the smocked graph
#         # find its neighboring underaly nodes
#         neigh_eid = sg.find_vtx_neighboring_edges(vid_sg)
#         trans = np.zeros((1,3))
#         count = 0
#         for eid in neigh_eid:
#             vtx = sg.E[eid, :]
#             vid = setdiff1d(vtx, vid_sg) # find the other endpoint
# #            if sg.is_vtx_underlay(vid):

#         '''


        
#         x0 = X_pleat.flatten()*scale

#         w_pleat_neq = 1e6
#         w_var = 1e1

#         start_time = time.time()
#         res_pleat = minimize(opti_energy_sg_pleat, 
#                        x0, 
#                        method='Nelder-Mead',
#                        args=(X_underlay, C_pleat_neq, w_pleat_neq, w_var, eps_neq), 
#                        options=opti_get_NelderMead_solver_options())
#         x0 = res_pleat.x
        
#         msg = 'optimization: embed the pleat graph: %f second' % (time.time() - start_time)
#         bpy.types.Scene.sl_props.runtime_log.append(msg)


#         print_runtime()
        
#         y1, y2, y3 = opti_energy_sg_pleat(res_pleat.x, X_underlay, C_pleat_neq, w_pleat_neq, w_var, eps_neq, True)

#         X_pleat = res_pleat.x.reshape(int(len(res_pleat.x)/3), 3)

#         print(y1, y2, y3)
        

#         X_all = np.concatenate((X_underlay, X_pleat), axis=0)

#         # print(X_all)
