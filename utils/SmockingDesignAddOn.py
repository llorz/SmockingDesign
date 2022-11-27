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
import math
from mathutils import Vector
import numpy as np
from bpy.types import Operator
from bpy.types import (Panel, Operator)
import os




# ========================================================================
#                          Global Variables
# ========================================================================

col_blue = (76/255.0, 201/255.0,240/255.0)
col_yellow = (254/255.0, 228/255.0, 64/255.0)
strokeSize = 10 # to plot the stiching lines


# global variables from user input

PROPS = [
    ('base_x', bpy.props.IntProperty(name='X', default=3, min=1, max=20)),
    ('base_y', bpy.props.IntProperty(name='Y', default=3, min=1, max=20)),
    ('num_x', bpy.props.FloatProperty(name='Xtile', default=3, min=1, max=20)),
    ('num_y', bpy.props.FloatProperty(name='Ytile', default=3, min=1, max=20)),
    ('shift_x', bpy.props.FloatProperty(name='Xshift', default=0, min=-10, max=10)),
    ('shift_y', bpy.props.FloatProperty(name='Yshift', default=0, min=-10, max=10)),
    ('type_tile', bpy.props.EnumProperty(items = [("regular", "regular", "tile in a regular grid"), ("radial", "radial", "tile in a radial grid")], name="Type", default="regular")),
    ('path_export', bpy.props.StringProperty(subtype='DIR_PATH', name='PATH')),
    ('filename_export', bpy.props.StringProperty(name='Name', default='my_pattern_name')),
    ('path_import', bpy.props.StringProperty(subtype='FILE_PATH', name='FILE'))]
    
    
# to extract the stitching lines from user selection

class StitchingLinesProp():
    currentDrawing = []
    # so we dont show multiple current drawings when repeated click "Done"
    if_curr_drawing_is_shown = False 
    if_user_is_drawing = True
    savedStitchingLines = []
    colSaved = col_blue
    colTmp = col_yellow


# data for optimization

class SolverData():
    unit_smocking_pattern = []
    full_smocking_pattern = []
    smocked_graph = []
    embeded_graph = []
    smocking_design = []    
    

# ========================================================================
#                         classes for the solver
# ========================================================================


class debug_clear(Operator):
    bl_idname = "object.debug_clear"
    bl_label = "clear data in scene"
    
    def execute(self, context):
        initialize()
            
        return {'FINISHED'}

class debug_print(Operator):
    bl_idname = "object.debug_print"
    bl_label = "print data in scene"
    
    def execute(self, context):
        
        props = bpy.types.Scene.sl_props
        dt = bpy.types.Scene.solver_data
        
        usp = dt.unit_smocking_pattern
        print('--------------------------------------------------------')
        print('data stored in current scene:')
        print('--------------------------------------------------------')
        if usp:
            print('Unit Smocking Pattern:')
            print('gridX: ' + str(usp.base_x) + ', gridY: ' + str(usp.base_y))        
            print('stitching points:')
            print(usp.stitching_points)
            
        else:
            print('Import the smokcing pattern first')
        
                
        return {'FINISHED'}







class debug_func(Operator):
    bl_idname = "object.debug_func"
    bl_label = "function to test"
    
    def execute(self, context):
        print('debugging...')
        
        props = bpy.types.Scene.sl_props
        dt = bpy.types.Scene.solver_data
        usp = dt.unit_smocking_pattern
        
        
        base_x = usp.base_x
        base_y = usp.base_y
        num_x = context.scene.num_x
        num_y = context.scene.num_y
        shift_x = context.scene.shift_x
        shift_y = context.scene.shift_y
        
        len_x = num_x * base_x - shift_x
        len_y = num_y * base_y - shift_y
        
        print(len_x, len_y)
        
        # create the full grid with size len_x by len_y
        gx, gy = create_grid(len_x, len_y)
        
        F, V, E = extract_graph_from_meshgrid(gx, gy, True)
        construct_object_from_mesh_to_scene(V, F, 'FullPattern', 'SmockingPattern')
        
        mesh = bpy.data.objects['FullPattern']
        mesh.scale = (1, 1, 1)
        mesh.location = (0, -len_y-2.5, 0)
        mesh.show_axis = False
        mesh.show_wire = True
        mesh.display_type = 'WIRE'
        select_one_object(mesh)
        
        # add annotation to full pattern
        add_text_to_scene(body="Full Smocking Pattern", 
                          location=(0,-2, 0), 
                          scale=(1,1,1),
                          obj_name='pattern_annotation',
                          coll_name='SmockingPattern')
        
        # tile the stitching lines from usp
        
        all_sp = [] # all stitching points
        all_sp_lid = [] # the stiching line ID
        all_sp_pid = [] # the stitching patch ID
        all_sp_vid = [] # the stitching pointID in the grid (pattern) mesh
        
        # extract each stitching lines from the unit pattern
        unit_lines = []
        for lid in range(len(usp.stitching_lines)):
            pos = usp.get_pts_in_stitching_line(lid)
            unit_lines.append(pos)    
        print(unit_lines)
        
        
        pid = 0
        lid = 0
        
        for ix in range( int(np.ceil(num_x)) + 1 ):
            for iy in range( int(np.ceil(num_y)) + 1):
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
                        
                        for ii in range(num):
                            vid = find_matching_rowID(V, new_line[ii,0:2])
                            all_sp_vid.append(vid[0][0].tolist())
                        
                        lid += 1
                
                pid += 1
                    
        print(usp.stitching_points)
        
#        print(all_sp)
        print(all_sp_lid)
        print(all_sp_pid)
        print(all_sp_vid)
        
#        print(V)
        
        SP = SmockingPattern(V, F, E, len_x, len_y, 
                             all_sp, 
                             all_sp_vid, 
                             all_sp_lid,
                             all_sp_pid)
        SP.plot()
        

        return {'FINISHED'}


    
    

# ========================================================================
#                         classes for the solver
# ========================================================================

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
    
    def plot(self):
        obj = generate_grid_for_unit_pattern(self.base_x, self.base_y)
        
        add_text_to_scene(body="Unit Smocking Pattern", 
                          location=(0, self.base_y+0.5, 0), 
                          scale=(1,1,1),
                          obj_name='grid_annotation',
                          coll_name='UnitSmockingPattern')
        
        for lid in range(len(self.stitching_lines)):
            pos = self.get_pts_in_stitching_line(lid)
            draw_stitching_line(pos, col_blue, "stitching_line_" + str(lid), strokeSize, 'UnitStitchingLines')
        






class SmockingPattern():
    """Full Smocking Pattern"""
    def __init__(self, V, F, E,
                 len_x, len_y,
                 stitching_points,
                 stitching_points_line_id,
                 stitching_points_patch_id):
        # the mesh for the smocking pattern
        self.V = V
        self.F = F
        self.E = E
        
        # for visualization
        self.len_x = len_x
        self.len_y = len_y
        
        # the stitching points: in 2D/3D positions
        self.stitching_points = np.array(stitching_points)
        # the lineID of the stitching points
        # the points with the same line ID will be sew together
        self.stitching_points_line_id = np.array(stitching_points_line_id)
        
        # the patchID of the stitching points from the tiling process
        # save this information for visualization only
        # not useful for optimization
        self.stitching_points_patch_id = np.array(stitching_points_patch_id)
        
        # the vtxID of each stitching points in V
        self.get_stitching_points_vtx_id()
    
    def get_stitching_points_vtx_id(self):
        all_sp_vid = []
        for ii in range(len(self.stitching_points)):
            vid = find_matching_rowID(self.V, self.stitching_points[ii,0:2])
            all_sp_vid.append(vid[0][0].tolist())
        self.stitching_points_vtx_id = np.array(all_sp_vid)
        
        
        
    def get_pts_in_stitching_line(self, lid):
    
        pids = find_index_in_list(self.stitching_points_line_id, lid)
        pos = np.array(self.stitching_points)[pids, :]
        
        return pos, pids    
   
   
    
    def plot(self):
        clean_objects_in_collection('SmockingPattern')
        
        construct_object_from_mesh_to_scene(self.V, self.F, 'FullPattern', 'SmockingPattern')
        mesh = bpy.data.objects['FullPattern']
        
        mesh.scale = (1, 1, 1)
        mesh.location = (0, -self.len_y-2.5, 0)
        mesh.show_axis = False
        mesh.show_wire = True
        mesh.display_type = 'WIRE'
        select_one_object(mesh)
        
        # add annotation to full pattern
        add_text_to_scene(body="Full Smocking Pattern", 
                          location=(0,-2, 0), 
                          scale=(1,1,1),
                          obj_name='pattern_annotation',
                          coll_name='SmockingPattern')
       
        clean_objects_in_collection('StitchingLines')       
        # visualize all stitching lines
        for lid in range(max(self.stitching_points_line_id)+1):
            
            # cannot use the position from the V, since the mesh is translated
            _, pids = self.get_pts_in_stitching_line(lid)
            print(pids)
            vtxID = self.stitching_points_vtx_id[pids]
            pos = get_vtx_pos(mesh, np.array(vtxID))
            
            # draw the stitching lines in the world coordinate
            draw_stitching_line(pos, col_blue, "stitching_line_" + str(lid), strokeSize, 'StitchingLines')
        

    
# ========================================================================
#                          Utility Functions
# ========================================================================

def delete_collection():    
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


def delete_all():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    
def clean_objects() -> None:
    for item in bpy.data.objects:
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
                      coll_name="SmockingPattern"):
                          
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


def get_vtx_pos(mesh, vids):
    p = [];
    for vid in vids:
        p.append(get_curr_vtx_pos(mesh, vid))
    return p


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

def create_line_stroke_from_gpencil(name="GPencil", line_width=12, coll_name='SmockingPattern'):
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
    

def draw_stitching_line(pts, col, name="stitching_line", line_width=12, coll_name='SmockingPattern'): 
    gpencil, gp_stroke = create_line_stroke_from_gpencil(name, line_width, coll_name)
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


def draw_saved_stitching_lines(context, coll_name='SmockingPattern'):
    props = bpy.context.scene.sl_props
    print(props.savedStitchingLines)
    for i in range(len(props.savedStitchingLines)):
        vids = props.savedStitchingLines[i]
        obj = bpy.data.objects['Grid']
        pts = get_vtx_pos(obj, vids)
        draw_stitching_line(pts, props.colSaved, "stitching_line_" + str(i), strokeSize, coll_name)
        
        
            
# ---------------------END: add strokes via gpencil ------------------------

            
def construct_object_from_mesh_to_scene(V, F, mesh_name, coll_name='SmockingPattern'):
    # input: V nv-by-2(3) array, F list of array
    
    # convert F into a list of list
    faces = F
    # convert V into a list of Vector()
    verts = [Vector((v[0], v[1], v[2] if len(v) > 2 else 0)) for v in V]
    
    # create mesh in blender
    mesh = bpy.data.meshes.new(mesh_name)
    mesh.from_pydata(verts, [], faces)
    mesh.update(calc_edges=False) # we use our own edgeID
    object = bpy.data.objects.new(mesh_name, mesh)
    # link the object to the scene
    bpy.context.scene.collection.children[coll_name].objects.link(object)
    


def generate_grid_for_unit_pattern(base_x, base_y, if_add_diag=False):
    # there is some but in blender subdivison
    # instead, manually create a grid
    
    gx, gy = create_grid(base_x, base_y)
    F, V, _ = extract_graph_from_meshgrid(gx, gy, if_add_diag)
    construct_object_from_mesh_to_scene(V, F, 'Grid', 'UnitSmockingPattern')
    

    mesh = bpy.data.objects['Grid']
    mesh.scale = (1, 1, 1)
    mesh.show_axis = False
    mesh.show_wire = True
    mesh.display_type = 'WIRE'
    select_one_object(mesh)
    
    return mesh
    

def generate_grid_for_full_pattern(len_x, len_y, if_add_diag=True):
    # create the full grid with size len_x by len_y
    gx, gy = create_grid(len_x, len_y)
    
    F, V, E = extract_graph_from_meshgrid(gx, gy, if_add_diag)
    construct_object_from_mesh_to_scene(V, F, 'FullPattern', 'SmockingPattern')
    
    mesh = bpy.data.objects['FullPattern']
    mesh.scale = (1, 1, 1)
    mesh.location = (0, -len_y-2.5, 0)
    mesh.show_axis = False
    mesh.show_wire = True
    mesh.display_type = 'WIRE'
    select_one_object(mesh)
                
    return mesh, F, V, E
    
        

def find_matching_rowID(V, pos):
    # V: a list of vtx positions
    # pos: the query position
    ind = np.where(np.all(np.array(V) == np.array(pos), axis=1))
    return ind


    
def initialize():
    delete_collection()
    clean_objects()
    
    # collections for unit smocking pattern
    my_coll = bpy.data.collections.new('UnitSmockingPattern')
    bpy.context.scene.collection.children.link(my_coll)
    
    my_coll_strokes = bpy.data.collections.new('UnitStitchingLines')
    my_coll.children.link(my_coll_strokes)
    
    # collections for the full smocking pattern
    
    my_coll = bpy.data.collections.new('SmockingPattern')
    bpy.context.scene.collection.children.link(my_coll)
    
    my_coll_strokes = bpy.data.collections.new('StitchingLines')
    my_coll.children.link(my_coll_strokes)
    

# ========================================================================
#                      Core functions for smocking pattern
# ========================================================================

    

def sort_edge(edges):
    # return the unique edges
    e_sort = [np.array([min(e), max(e)]) for e in edges]
    e_unique = np.unique(e_sort, axis = 0)
    return e_unique




def create_grid(num_x, num_y):
#        scale = max(base_x, base_y)
#        # subdivison a bit buggy here
#        bpy.ops.mesh.primitive_grid_add(size=scale, 
#                                        location=(base_x/2, base_y/2,0),
#                                        x_subdivisions=base_x + 1,
#                                        y_subdivisions=base_y + 1) 

    # note num_x/num_y can be float :/
#    xx = range(num_x+1) # [0, ..., num_x]
#    yy = range(num_y+1) # [0, ..., num_y]    
    
    
    xx = list(range(int(np.floor(num_x))+1)) + [num_x]
    yy = list(range(int(np.floor(num_y))+1)) + [num_y]
    
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
        print('Add diagonal edges')
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
    
    for ix in range( int(np.ceil(num_x)) + 1 ):
        for iy in range( int(np.ceil(num_y)) + 1):
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
        




# ========================================================================
#                          Functions for UIs
# ========================================================================

# ------------------------------------------------------------------------
#    Drawing the Unit Pattern using mouse
# ------------------------------------------------------------------------

from bpy.props import IntProperty, FloatProperty

class SelectStitchingPoint(Operator):
    """Draw a stitching line by selecting points in order"""
    bl_idname = "object.modal_operator"
    bl_label = "Simple Modal Operator"
   
    def modal(self, context, event):
        
        props = bpy.context.scene.sl_props
        
        if props.if_user_is_drawing:
        
            if event.type == 'LEFTMOUSE':

                print('I hit the left mouse')
                
                obj = bpy.data.objects['Grid']
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
        context.window_manager.drawing_started = True
        
        

        if props.if_user_is_drawing:
            props.currentDrawing = []
            bpy.ops.object.mode_set(mode = 'EDIT') 
            
            obj = bpy.data.objects['Grid']
            deselect_all_vert_in_mesh(obj)
            
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            self.report({'WARNING'}, "No active object, could not finish")
            return {'CANCELLED'}

        
class FinishCurrentDrawing(Operator):
    """Finish drawing the current stitching line"""
    bl_idname = "edit.finish_drawing"
    bl_label = "Finish drawing the current stitching line"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        
        bpy.ops.object.mode_set(mode = 'OBJECT')
        
        props = bpy.context.scene.sl_props
        
        if not props.if_curr_drawing_is_shown:
            mesh = bpy.data.objects['Grid']
            pts = get_vtx_pos(mesh, props.currentDrawing)
            draw_stitching_line(pts, props.colTmp, "stitching_line_tmp", strokeSize, 'UnitStitchingLines')
        
            props.if_curr_drawing_is_shown = True

        props.if_user_is_drawing = False
        context.window_manager.drawing_started = False

        return {'FINISHED'}


class FinishPattern(Operator):
    """Finish creating the unit smocking pattern"""
    bl_idname = "edit.finish_creating_pattern"
    bl_label = "Finish creating the unit smocking pattern"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        
        props = bpy.context.scene.sl_props
                
        mesh = bpy.data.objects['Grid']
        
        usp = get_usp_from_saved_stitching_lines(props.savedStitchingLines, 
                                                 mesh, 
                                                 context.scene.base_x,
                                                 context.scene.base_y)

        delete_all()                               
        usp.plot()
        
        # save the loaded pattern to the scene
        bpy.types.Scene.solver_data.unit_smocking_pattern = usp
        
        return {'FINISHED'}



class SaveCurrentStitchingLine(Operator):
    """Save this stitching line"""
    bl_idname = "edit.save_current_stitching_line"
    bl_label = "Add one stiching line"
    bl_options = {'REGISTER', 'UNDO'}
        
    def execute(self, context):
        props = bpy.context.scene.sl_props
        
        if props.currentDrawing:
            props.savedStitchingLines.append(props.currentDrawing)
        
        props.currentDrawing = []
        
        print(props.savedStitchingLines)
        
        clean_objects_in_collection('UnitStitchingLines')
        
        draw_saved_stitching_lines(context, 'UnitStitchingLines')
        
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
                                                 bpy.data.objects['Grid'], 
                                                 context.scene.base_x,
                                                 context.scene.base_y)
        write_usp(file_name, usp)
#        
#        write_usp(file_name, 
#                  context.scene.base_x, 
#                  context.scene.base_y, 
#                  props.savedStitchingLines, 
#                  bpy.data.objects['Grid'])
                  
        
        return {'FINISHED'}
    



class ImportUnitPattern(Operator):
    """Import an existing unit smocking pattern"""
    bl_idname = "object.import_unit_pattern"
    bl_label = "Import a unit pattern"
    
    def execute(self, context):
        # refresh the drawing of the unit pattern
        initialize()
        
        file_name = bpy.path.abspath(context.scene.path_import)
        usp = read_usp(file_name)
        
        usp.plot()
        # save the loaded pattern to the scene
        bpy.types.Scene.solver_data.unit_smocking_pattern = usp
        
        return {'FINISHED'}
        
    



# ------------------------------------------------------------------------
#    Create Unit Grid
# ------------------------------------------------------------------------

class CreateGrid(Operator):
    """Create a grid for specifying unit smocking pattern"""
    bl_idname = "object.create_grid"
    bl_label = "Create Grid"
    bl_options = {'REGISTER', 'UNDO'}

    

    def execute(self, context):
        clean_objects()
        base_x = context.scene.base_x
        base_y = context.scene.base_y
        
        generate_grid_for_unit_pattern(base_x, base_y)
        
        add_text_to_scene(body="Unit Smocking Pattern", 
                          location=(0, base_y+0.5, 0), 
                          scale=(1,1,1),
                          obj_name='grid_annotation',
                          coll_name='UnitSmockingPattern')
                          
        # clear the old drawings
        props = bpy.types.Scene.sl_props
        props.savedStitchingLines = []
        props.currentDrawing = []
        
        return {'FINISHED'}



# ------------------------------------------------------------------------
#    Full Smocking Pattern by Tiling
# ------------------------------------------------------------------------
class FullSmockingPattern(Operator):
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
        
        mesh, F, V, E = generate_grid_for_full_pattern(len_x, len_y, True)
    
        SP = SmockingPattern(V, F, E, len_x, len_y, 
                             all_sp, 
                             all_sp_lid,
                             all_sp_pid)
        SP.plot()
        
        # save the loaded pattern to the scene
        bpy.types.Scene.solver_data.full_smocking_pattern = SP
        
    
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
    bl_options ={"HEADER_LAYOUT_EXPAND"}
    
    def draw(self, context):
        
        row = self.layout.row()
        row.operator(debug_clear.bl_idname, text="clear everything", icon='QUIT')
        row = self.layout.row()
        row.operator(debug_print.bl_idname, text="print data in scene", icon='GHOST_ENABLED')
        row = self.layout.row()
        row.operator(debug_func.bl_idname, text="test function", icon="GHOST_DISABLED")



class UnitGrid_panel:
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SmockingDesign"
    bl_options ={"HEADER_LAYOUT_EXPAND"}
     
    

class UNITGRID_PT_main(UnitGrid_panel, bpy.types.Panel):
    bl_label = "Unit Smocking Pattern"
    bl_idname = "SD_PT_unit_grid_main"
#    bl_options ={"HEADER_LAYOUT_EXPAND"}
            
    def draw(self, context):
        pass

    
class UNITGRID_PT_create(UnitGrid_panel, bpy.types.Panel):
    bl_parent_id = 'SD_PT_unit_grid_main'
    bl_label = "Create a New Smocking Pattern"
#    bl_options ={"DEFAULT_CLOSED"}
    
    
    def draw(self, context):
        props = context.scene.sl_props
        layout = self.layout
        
        layout.label(text= "Generate A Grid for Drawing:")
        row = layout.row()
        row.prop(context.scene,'base_x')
        row.prop(context.scene,'base_y')

        
        row = layout.row()
        row.operator(CreateGrid.bl_idname, text="Generate Grid", icon='GRID')

        
        row = layout.row()
        layout.label(text= "Draw A Stitching Line")
        row = layout.row()
        
        if(not context.window_manager.drawing_started):
            row.operator(SelectStitchingPoint.bl_idname, text="Draw", icon='GREASEPENCIL')
        else:
            row.operator(FinishCurrentDrawing.bl_idname, text="Done", icon='CHECKMARK')
       
        row.operator(SaveCurrentStitchingLine.bl_idname, text="Add", icon='ADD')
        
        row = layout.row()
        row.operator(FinishPattern.bl_idname, text="Finish Unit Pattern Design", icon='FUND')
        
         
        row = layout.row()
        row = layout.row()        
        layout.label(text= "Export the Created Unit Smocking Pattern")
        row = layout.row()  
        row.prop(context.scene, 'path_export')
        row = layout.row()  
        row.prop(context.scene, 'filename_export')
        row = layout.row()  
        row.operator(ExportUnitPattern.bl_idname, text="Export", icon='EXPORT')
        
                 

class UNITGRID_PT_load(UnitGrid_panel, bpy.types.Panel):
    bl_parent_id = 'SD_PT_unit_grid_main'
    bl_label = "Load Existing Smocking Pattern"
    bl_options ={"DEFAULT_CLOSED"}
    
    def draw(self, context):
        layout = self.layout
        layout.label(text= "Load an Existing Pattern")
        row = layout.row()
        row.prop(context.scene, 'path_import')
        row = layout.row()  
        row.operator(ImportUnitPattern.bl_idname, text="Import", icon='IMPORT')

        
class FullGrid_panel(bpy.types.Panel):
    bl_label = "Full Smocking Pattern"
    bl_idname = "SD_PT_full_grid"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SmockingDesign"
    bl_options ={"DEFAULT_CLOSED"}
    
    def draw(self, context):
        
        layout = self.layout
        layout.label(text= "Tiling Parameters:")
        row = layout.row()
        row.prop(context.scene,'num_x')
        row.prop(context.scene,'num_y')
        row = layout.row()
        row.prop(context.scene,'shift_x')
        row.prop(context.scene,'shift_y')
        row = layout.row()
        row.prop(context.scene, 'type_tile')
        
        row = layout.row()
        row.operator(FullSmockingPattern.bl_idname, text="Generate by Tiling", icon='FILE_VOLUME')
        
        
        
# ========================================================================
#                          Registration
# ========================================================================

wm = bpy.types.WindowManager
wm.drawing_started = bpy.props.BoolProperty(default=False)


_classes = [
    CreateGrid,
    SaveCurrentStitchingLine,
    SelectStitchingPoint,
    FinishCurrentDrawing,
    FinishPattern,
    
    debug_clear,
    debug_print,
    debug_func,
#    UnitSmockingPattern,
    
    FullSmockingPattern,
    
    ExportUnitPattern,
    ImportUnitPattern,
    
    debug_panel,
    
    UNITGRID_PT_main,
    UNITGRID_PT_create,
    UNITGRID_PT_load,
    
    FullGrid_panel
 ]


def register():
    for (prop_name, prop_value) in PROPS:
        setattr(bpy.types.Scene, prop_name, prop_value)
        
    for cls in _classes:
        bpy.utils.register_class(cls)
        
    bpy.types.Scene.sl_props = StitchingLinesProp()
    bpy.types.Scene.solver_data = SolverData()
    

def unregister():
    for cls in _classes:
        bpy.utils.unregister_class(cls)
        
    del bpy.types.Scene.sf_props    
    del bpy.tpyes.Scene.solver_data

if __name__ == "__main__":
    register()
#    bpy.ops.object.export_unit_pattern('INVOKE_DEFAULT')




# ========================================================================
#                         as useless as jing
# ========================================================================


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
#        mesh = bpy.data.objects['Grid']
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
