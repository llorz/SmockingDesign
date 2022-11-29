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
    # for unit grid
    ('base_x', bpy.props.IntProperty(name='X', default=3, min=1, max=20)),
    ('base_y', bpy.props.IntProperty(name='Y', default=3, min=1, max=20)),
    # import/export stitching lines
    ('path_export', bpy.props.StringProperty(subtype='DIR_PATH', name='Path',default='/tmp/')),
    ('filename_export', bpy.props.StringProperty(name='Name', default='my_pattern_name')),
    ('path_import', bpy.props.StringProperty(subtype='FILE_PATH', name='File')),
    # for full grid (full smocking pattern)
    ('num_x', bpy.props.FloatProperty(name='Xtile', default=3, min=1, max=20)),
    ('num_y', bpy.props.FloatProperty(name='Ytile', default=3, min=1, max=20)),
    ('shift_x', bpy.props.IntProperty(name='Xshift', default=0, min=-10, max=10)),
    ('shift_y', bpy.props.IntProperty(name='Yshift', default=0, min=-10, max=10)),
    ('type_tile', bpy.props.EnumProperty(items = [("regular", "regular", "tile in a regular grid"), ("radial", "radial", "tile in a radial grid")], name="Type", default="regular")),
    ('margin_top', bpy.props.FloatProperty(name='Top', default=0, min=0, max=10)),
    ('margin_bottom', bpy.props.FloatProperty(name='Bottom', default=0, min=0, max=10)),
    ('margin_left', bpy.props.FloatProperty(name='Left', default=0, min=0, max=10)),
    ('margin_right', bpy.props.FloatProperty(name='Right', default=0, min=0, max=10)),
    # export the full smocking pattern as obj
    ('path_export_fullpattern', bpy.props.StringProperty(subtype='FILE_PATH', name='Path', default='/tmp/')),
    ('filename_export_fullpattern', bpy.props.StringProperty(name='Name', default='my_pattern_name')),
    ('export_format', bpy.props.EnumProperty(items = [(".obj", "OBJ", ".obj"), (".off", "OFF", ".off")], name="Format", default=".obj")),
    # FSP: combine two patterns
    ('file_import_p1', bpy.props.StringProperty(subtype='FILE_PATH', name='P1')),
    ('file_import_p2', bpy.props.StringProperty(subtype='FILE_PATH', name='P2')),
    ('combine_direction', bpy.props.EnumProperty(items = [("x", "x", "x"), ("y", "y", "y")], name="Axis", default="x")),
    ('combine_space', bpy.props.FloatProperty(name='space', default=2, min=1, max=20))
        ]
    
    
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

    # temporary data
    tmp_fsp1 = []
    tmp_fsp2 = []
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



class debug_func(Operator):
    bl_idname = "object.debug_func"
    bl_label = "function to test"
    
    def execute(self, context):
        print('debugging...')
        
         
        
        my_coll = bpy.data.collections.new('TmpCollection')
        bpy.context.scene.collection.children.link(my_coll)
    
        # load the exiting smocking pattern to my_coll
        file_name = context.scene.file_import_p1
        fsp = read_obj_to_fsp(file_name)
        fsp.plot(location=[-5,-5])


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
#            add_stroke_to_gpencil(pos, col_blue, "USP_StitchingLines", strokeSize)

    def info(self):
        print('------------------------------')
        print('Unit Smocking Pattern:')
        print('------------------------------')
        print('base_x: ' + str(self.base_x) + ', base_y: ' + str(self.base_y))
        print('No. stitching lines: ' + str(len(self.stitching_lines)))



class SmockingPattern():
    """Full Smocking Pattern"""
    def __init__(self, V, F, E,
                 stitching_points,
                 stitching_points_line_id,
                 stitching_points_patch_id=None,
                 stitching_points_vtx_id=None,
                 pattern_name = "FllPattern", 
                 coll_name='SmockingPattern',
                 stroke_coll_name = "StitchingLines",
                 annotation_text="Full Smocking Pattern"):
        # the mesh for the smocking pattern
        self.V = V # can be 2D or 3D vtx positions
        self.F = F
        self.E = E
        
        # the stitching points: in 2D/3D positions
        self.stitching_points = np.array(stitching_points)
        # the lineID of the stitching points
        # the points with the same line ID will be sew together
        self.stitching_points_line_id = np.array(stitching_points_line_id)
        
        # the patchID of the stitching points from the tiling process
        # save this information for visualization only
        # not useful for optimization
        if stitching_points_patch_id:
            self.stitching_points_patch_id = np.array(stitching_points_patch_id)
        else: # do not have it, we then use the line_id
            self.stitching_points_patch_id = np.array(stitching_points_line_id)
        
        # the vtxID of each stitching points in V
        if stitching_points_vtx_id:
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
            vid = find_matching_rowID(self.V, self.stitching_points[ii,0:len(self.V[0])])
            
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
        
        clean_objects_in_collection(self.coll_name)
        
        construct_object_from_mesh_to_scene(self.V, self.F, self.pattern_name, self.coll_name)

        mesh = bpy.data.objects[self.pattern_name]
        
        mesh.scale = (1, 1, 1)
        mesh.location = (location[0]-min(self.V[:,0]), location[1]-max(self.V[:,1])-2.5, 0)
        mesh.show_axis = False
        mesh.show_wire = True
        mesh.display_type = 'WIRE'
        select_one_object(mesh)
        
        # add annotation to full pattern
        add_text_to_scene(body=self.annotation_text, 
                          location=(location[0], location[1]-2, 0), 
                          scale=(1,1,1),
                          obj_name=self.pattern_name+"_annotation",
                          coll_name=self.coll_name)
       
        # visualize all stitching lines
        clean_objects_in_collection(self.stroke_coll_name)

        for lid in range(max(self.stitching_points_line_id)+1):
            
            # cannot use the position from the V, since the mesh is translated
#            _, pids = self.get_pts_in_stitching_line(lid)
##            print(pids)
#            vtxID = self.stitching_points_vtx_id[pids]
#            
            vtxID = self.get_vid_in_stitching_line(lid)
            
            pos = get_vtx_pos(mesh, np.array(vtxID))
            
            # draw the stitching lines in the world coordinate
            draw_stitching_line(pos, col_blue, "stitching_line_" + str(lid), strokeSize, self.stroke_coll_name)
#            add_stroke_to_gpencil(pos, col_blue, "FSP_StitchingLines", strokeSize)
            
            
    
    def info(self):
        print('------------------------------')
        print('Full Smocking Pattern:')
        print('------------------------------')
        print('No. vertices: ' + str(len(self.V)))
        print('No. faces: ' + str(len(self.F)))
        print('No. stitching lines: ' + str(max(self.stitching_points_line_id)+1))
        print('No. unit patches: ' + str(max(self.stitching_points_patch_id)+1))
        print(self.stitching_points_patch_id)

    
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


def delete_one_collection(coll_name):
    for coll in bpy.data.collections:
        if coll_name in coll.name:
            for child in coll.children:
                bpy.data.collections.remove(child)
                clean_objects_in_collection(coll.name)




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
            f.write("e %d %d\n" % (fsp.E[e_id, 0] + 1, fsp.E[e_id, 1] + 1))
    
        # add stiching lines to the object
        for lid in range(fsp.num_stitching_lines()):
            vtxID = fsp.get_vid_in_stitching_line(lid)
            
            for ii in range(len(vtxID)-1):
                # f.write('l ' + str(vtxID[ii]+1) + ' ' + str(vtxID[ii+1]+1) + '\n')
                f.write('l %d %d\n' % (vtxID[ii] + 1 , vtxID[ii+1] + 1))
                

def read_obj_to_fsp(file_name, 
                    pattern_name = "FllPattern", 
                    coll_name='SmockingPattern',
                    stroke_coll_name = "StitchingLines",
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

    fsp = SmockingPattern(V, F, E, 
                          all_sp, all_sp_lid, None, all_sp_vid,
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
    construct_object_from_mesh_to_scene(V, F, 'Grid', 'UnitSmockingPattern')
    
    mesh = bpy.data.objects['Grid']
    
    show_mesh(mesh)
    
    
    return mesh
    

def generate_tiled_grid_for_full_pattern(len_x, len_y, if_add_diag=True):
    # create the full grid with size len_x by len_y
    gx, gy = create_grid(len_x, len_y)
    
    F, V, E = extract_graph_from_meshgrid(gx, gy, if_add_diag)
    
    construct_object_from_mesh_to_scene(V, F, 'FullPattern', 'SmockingPattern')
    
    mesh = bpy.data.objects['FullPattern']
    
    show_mesh(mesh, 
              scale=(1,1,1), 
              location=(0, -len_y - 2.5, 0))
    
                
    return mesh, F, V, E
    
        

def find_matching_rowID(V, pos):
    # V: a list of vtx positions
    # pos: the query position
    ind = np.where(np.all(np.array(V) == np.array(pos), axis=1))
    return ind


        

def refresh_stitching_lines():  
    all_sp = []
    all_sp_lid = []
    lid = 0
    
    trans = get_translation_of_mesh('FullPattern') 
    
    # check the remaining stitching lines
    saved_sl_names = []
    for obj in bpy.data.collections['StitchingLines'].objects:
        saved_sl_names.append(obj.name)
#    print(saved_sl_names)
    
    for sl_name in saved_sl_names:
        gp = bpy.data.grease_pencils[sl_name]
        pts = gp.layers.active.active_frame.strokes[0].points

        for p in pts:
            # p is in the world coordinate
            # need to translate it back to the grid-coordinate
            all_sp.append([p.co[0]-trans[0], p.co[1]-trans[1], p.co[2]-trans[2]])
            all_sp_lid.append(lid)
        lid += 1
        

    all_sp_pid = all_sp_lid # now patchID is useless
    
    return all_sp, all_sp_lid, all_sp_pid




    
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
    
    
    # collections for the smocked graph
    my_coll = bpy.data.collections.new('SmockedGraph')
    bpy.context.scene.collection.children.link(my_coll)
    
    
    
# ========================================================================
#                      Core functions for smocking pattern
# ========================================================================

    

def sort_edge(edges):
    # return the unique edges
    e_sort = [np.array([min(e), max(e)]) for e in edges]
    e_unique = np.unique(e_sort, axis = 0)
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
        context.window_manager.usp_drawing_started = True
        
        

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

        
class USP_FinishCurrentDrawing(Operator):
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
        context.window_manager.usp_drawing_started = False

        return {'FINISHED'}


class USP_FinishPattern(Operator):
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

class USP_CreateGrid(Operator):
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
    
        SP = SmockingPattern(V, F, E,
                             all_sp, 
                             all_sp_lid,
                             all_sp_pid)
        SP.plot()
        
        # save the loaded pattern to the scene
        bpy.types.Scene.solver_data.full_smocking_pattern = SP
        
    
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
                                    m_top, m_bottom, m_left, m_right)
    
        F, V, E = extract_graph_from_meshgrid(gx, gy, True)
        
        fsp.update_mesh(V, F, E)
        fsp.plot()
        
        
        return {'FINISHED'}


class FSP_Export(Operator):
    bl_idname = "object.full_smocking_pattern_export"
    bl_label = "Export the Smocking Pattern to A Mesh"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        
        dt = bpy.types.Scene.solver_data
        
        save_dir = bpy.path.abspath(context.scene.path_export_fullpattern)
        file_name = context.scene.filename_export_fullpattern + context.scene.export_format
        
        fsp = dt.full_smocking_pattern
        filepath = save_dir + file_name
        write_fsp_to_obj(fsp, filepath)

        return {'FINISHED'}
    

class FSP_DeleteStitchingLines_start(Operator):
    bl_idname = "object.fsp_delete_stitching_lines_start"
    bl_label = "Delete stitching lines start"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        bpy.ops.object.select_all(action='DESELECT')
        
        # select all stitching lines in the full pattern
        for obj in bpy.data.collections['StitchingLines'].objects:
            obj.select_set(True)
            
        return {'FINISHED'}


    
    

class FSP_DeleteStitchingLines_done(Operator):
    bl_idname = "object.fsp_delete_stitching_lines_done"
    bl_label = "Delete stitching lines done"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
         
        dt = bpy.types.Scene.solver_data
        fsp = dt.full_smocking_pattern
        
        
        all_sp, all_sp_lid, all_sp_pid = refresh_stitching_lines()
        
        fsp.update_stitching_lines(all_sp, all_sp_lid, all_sp_pid)
        
        fsp.plot()
                    
        return {'FINISHED'}




class FSP_AddStitchingLines_draw_start(Operator):
    bl_idname = "object.fsp_add_stitching_lines_draw_start"
    bl_label = "Start drawing a new stitching line"
    bl_options = {'REGISTER', 'UNDO'}
    
    def modal(self, context, event):
        
        props = bpy.context.scene.sl_props
        
        if props.if_user_is_drawing:
        
            if event.type == 'LEFTMOUSE':

                obj = bpy.data.objects['FullPattern']
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
        for obj in bpy.data.collections['StitchingLines'].objects:
            if 'tmp' in obj.name:
                bpy.data.objects.remove(obj)
         
        props.if_user_is_drawing = True
        props.if_curr_drawing_is_shown = False
        context.window_manager.fsp_drawing_started = True
        

        if props.if_user_is_drawing:
            props.currentDrawing = []
            bpy.ops.object.mode_set(mode = 'EDIT') 
            
            obj = bpy.data.objects['FullPattern']
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
            mesh = bpy.data.objects['FullPattern']
            pts = get_vtx_pos(mesh, props.currentDrawing)
            draw_stitching_line(pts, props.colTmp, "stitching_line_tmp", strokeSize, 'StitchingLines')
        
            props.if_curr_drawing_is_shown = True

        props.if_user_is_drawing = False
        context.window_manager.fsp_drawing_started = False
        return {'FINISHED'}




class FSP_AddStitchingLines_draw_add(Operator):
    bl_idname = "object.fsp_add_stitching_lines_draw_add"
    bl_label = "Add this new stitching line"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        
        for obj in bpy.data.collections['StitchingLines'].objects:
            if 'tmp' in obj.name:
                new_name = "new_stitching_line_" + str(len(bpy.data.collections['StitchingLines'].objects)-1)
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
        
        all_sp, all_sp_lid, all_sp_pid = refresh_stitching_lines()
        
        fsp.update_stitching_lines(all_sp, all_sp_lid, all_sp_pid)
        
        fsp.plot()
                    
        return {'FINISHED'}



class FSP_EditMesh_start(Operator):
    bl_idname = "object.fsp_edit_mesh_start"
    bl_label = "Edit the mesh of the smocking pattern"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        
        bpy.ops.object.select_all(action='DESELECT')
        
        bpy.data.objects['FullPattern'].select_set(True)
        
        bpy.ops.object.mode_set(mode = 'EDIT') 
        
        return {'FINISHED'}
    
    
class FSP_EditMesh_done(Operator):
    bl_idname = "object.fsp_edit_mesh_done"
    bl_label = "Finish the edits and update the smocking pattern"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        print('not done yet :/')
        return {'FINISHED'}




def clean_tmp_collections():
    
    if_tmp_coll_exist = False
    for coll in bpy.data.collections:
        if "TmpCollection" in coll.name:
            if_tmp_coll_exist = True

    if if_tmp_coll_exist:
        # remove all the existing objects
        clean_objects_in_collection('TmpCollection1')
        clean_objects_in_collection('TmpCollection2')
        clean_objects_in_collection('TmpStitchingLines1')
        clean_objects_in_collection('TmpStitchingLines2')
    else:
        # create new collections
        for cid in range(1,3): 
            coll_name = "TmpCollection" + str(cid)
            stroke_coll_name = "TmpStitchingLines" + str(cid)
            my_coll = bpy.data.collections.new(coll_name)
            bpy.context.scene.collection.children.link(my_coll)

            my_coll_strokes = bpy.data.collections.new(stroke_coll_name)
            my_coll.children.link(my_coll_strokes)
            
        


  

class FSP_CombinePatterns_load_first(Operator):
    bl_idname = "object.fsp_combine_patterns_load_first"
    bl_label = "Load the first pattern"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        coll_name = "TmpCollection1"
        stroke_coll_name = "TmpStitchingLines1"
        pattern_name = 'Pattern01'
        annotation_text = "Pattern 01"
        
        clean_tmp_collections()
    
        # load the exiting smocking pattern to my_coll
        file_name = context.scene.file_import_p1
        
        fsp = read_obj_to_fsp(file_name, pattern_name, coll_name, stroke_coll_name, annotation_text)

        dt = bpy.types.Scene.solver_data
        dt.tmp_fsp1 = fsp # save the data to scene

        tmp_fsp1_loc = [-fsp.return_pattern_width() - 2, fsp.return_pattern_height()+1]

        if dt.tmp_fsp2 != []:
            minx = -max(dt.tmp_fsp2.return_pattern_width(), fsp.return_pattern_width())-2
            tmp_fsp1_loc = [minx, fsp.return_pattern_height()+1]
            tmp_fsp2_loc = [minx, -2.5]
            dt.tmp_fsp2.plot(tmp_fsp2_loc)

        fsp.plot(tmp_fsp1_loc)

        return {'FINISHED'}



    
    
class FSP_CombinePatterns_load_second(Operator):
    bl_idname = "object.fsp_combine_patterns_load_second"
    bl_label = "Load the second pattern"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        coll_name = "TmpCollection2"
        stroke_coll_name = "TmpStitchingLines2"
        pattern_name = 'Pattern02'
        annotation_text = "Pattern 02"
        
        clean_tmp_collections()

    
        # load the exiting smocking pattern to my_coll
        file_name = context.scene.file_import_p2
        
        fsp = read_obj_to_fsp(file_name, pattern_name, coll_name, stroke_coll_name, annotation_text)

        dt = bpy.types.Scene.solver_data
        dt.tmp_fsp2 = fsp # save the data to scene
        tmp_fsp2_loc = [-fsp.return_pattern_width() - 2,  -2.5]

        if dt.tmp_fsp1 != []:
            minx = -max(dt.tmp_fsp1.return_pattern_width(), fsp.return_pattern_width())-2
            tmp_fsp1_loc = [minx, dt.tmp_fsp1.return_pattern_height()+1]
            tmp_fsp2_loc = [minx, -2.5]
            dt.tmp_fsp1.plot(tmp_fsp1_loc)

        fsp.plot(tmp_fsp2_loc)

        return {'FINISHED'}




class FSP_CombinePatterns(Operator):
    bl_idname = "object.fsp_combine_patterns"
    bl_label = "Combined two patterns"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        print('not done yet :/')
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
    bl_label = "Create a New Pattern"
#    bl_options ={"DEFAULT_CLOSED"}
    
    
    def draw(self, context):
        props = context.scene.sl_props
        layout = self.layout
        
        layout.label(text= "Generate A Grid for Drawing:")
        row = layout.row()
        row.prop(context.scene,'base_x')
        row.prop(context.scene,'base_y')

        
        row = layout.row()
        row.operator(USP_CreateGrid.bl_idname, text="Generate Grid", icon='GRID')

        
        row = layout.row()
        layout.label(text= "Draw A Stitching Line")
        row = layout.row()
        
        if(not context.window_manager.usp_drawing_started):
            row.operator(USP_SelectStitchingPoint.bl_idname, text="Draw", icon='GREASEPENCIL')
        else:
            row.operator(USP_FinishCurrentDrawing.bl_idname, text="Done", icon='CHECKMARK')
       
        row.operator(USP_SaveCurrentStitchingLine.bl_idname, text="Add", icon='ADD')
        
        row = layout.row()
        row.operator(USP_FinishPattern.bl_idname, text="Finish Unit Pattern Design", icon='FUND')
        
         
        row = layout.row()
        row = layout.row()        
        layout.label(text= "Export Current Unit Smocking Pattern")
        row = layout.row()  
        row.prop(context.scene, 'path_export')
        row = layout.row()  
        row.prop(context.scene, 'filename_export')
        row = layout.row()  
        row.operator(ExportUnitPattern.bl_idname, text="Export", icon='EXPORT')
        
                 

class UNITGRID_PT_load(UnitGrid_panel, bpy.types.Panel):
    bl_parent_id = 'SD_PT_unit_grid_main'
    bl_label = "Load Existing Pattern"
    bl_options ={"DEFAULT_CLOSED"}
    
    def draw(self, context):
        layout = self.layout
#        layout.label(text= "Load an Existing Pattern")
        row = layout.row()
        row.prop(context.scene, 'path_import')
        row = layout.row()  
        row.operator(ImportUnitPattern.bl_idname, text="Import", icon='IMPORT')


class FullGrid_panel():
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SmockingDesign"
    bl_options ={"HEADER_LAYOUT_EXPAND"}
     
    

class FULLGRID_PT_main(FullGrid_panel, bpy.types.Panel):
    bl_label = "Full Smocking Pattern"
    bl_idname = "SD_PT_full_grid_main"
#    bl_options ={"HEADER_LAYOUT_EXPAND"}
            
    def draw(self, context):
        pass



        
class FULLGRID_PT_tile(FullGrid_panel, bpy.types.Panel):
    bl_label = "Tile Unit Grid"
    bl_parent_id = 'SD_PT_full_grid_main'

    
    def draw(self, context):
        
        layout = self.layout
        layout.label(text= "Repeat the unit pattern:")
        row = layout.row()
        row.prop(context.scene,'num_x')
        row.prop(context.scene,'num_y')
        
        layout.label(text= "Shift the unit pattern:")
        row = layout.row()
        row.prop(context.scene,'shift_x')
        row.prop(context.scene,'shift_y')
        
        row = layout.row()
        row = layout.row()
        row.prop(context.scene, 'type_tile')
        
        row = layout.row()
        row = layout.row()
        row.operator(FSP_Tile.bl_idname, text="Generate by Tiling", icon='FILE_VOLUME')






class FULLGRID_PT_combine_patterns(FullGrid_panel, bpy.types.Panel):
    bl_label = "Combine Two Patterns"
    bl_parent_id = 'SD_PT_full_grid_main'
    
    def draw(self, context):
        layout = self.layout
        row = layout.row()
        row.prop(context.scene, 'file_import_p1')
        row.operator(FSP_CombinePatterns_load_first.bl_idname, text="Import", icon='IMPORT')
        
        row = layout.row()
        row.prop(context.scene, 'file_import_p2')
        row.operator(FSP_CombinePatterns_load_second.bl_idname, text="Import", icon='IMPORT')


        layout.label(text= "Parameters:")
        row = layout.row()
        row.prop(context.scene, 'combine_direction')
        row.prop(context.scene, 'combine_space')
        
        row = layout.row()
        row.operator(FSP_CombinePatterns.bl_idname, text="Combined Two Patterns", icon="NODE_COMPOSITING")







class FULLGRID_PT_edit_pattern(FullGrid_panel, bpy.types.Panel):
    bl_label = "Edit Current Pattern"
    bl_parent_id = 'SD_PT_full_grid_main'
    
    def draw(self, context):
        layout = self.layout
        layout.label(text= "Delete Stitching Lines")
        row = layout.row()
        row.operator(FSP_DeleteStitchingLines_start.bl_idname, text="Delete", icon="PANEL_CLOSE")
        row.operator(FSP_DeleteStitchingLines_done.bl_idname, text="Done", icon="CHECKMARK")
        
        
        layout.label(text= "Add New Stitching Lines")
        row = layout.row()
        if(not context.window_manager.fsp_drawing_started):
            row.operator(FSP_AddStitchingLines_draw_start.bl_idname, text="Draw", icon='GREASEPENCIL')
        else:
            row.operator(FSP_AddStitchingLines_draw_end.bl_idname, text="Done", icon='CHECKMARK')
        
        row.operator(FSP_AddStitchingLines_draw_add.bl_idname, text="Add", icon='ADD')
        row = layout.row()
        row.operator(FSP_AddStitchingLines_draw_finish.bl_idname, text="Finish Adding Extra Stitching Lines", icon='FUND')
        
        
        layout.label(text= "Edit the Smocking Grid")
        row = layout.row()
        row.operator(FSP_EditMesh_start.bl_idname, text="Edit", icon="EDITMODE_HLT")
        row.operator(FSP_EditMesh_done.bl_idname, text="Done", icon="CHECKMARK")
        
        
        

class FULLGRID_PT_add_margin(FullGrid_panel, bpy.types.Panel):
    bl_label = "Add Margin to Current Pattern"
    bl_parent_id = 'SD_PT_full_grid_main'

    
    def draw(self, context):
        
        layout = self.layout
        row = layout.row()
        row.prop(context.scene,'margin_top')
        row.prop(context.scene,'margin_bottom')
        row = layout.row()
        row.prop(context.scene,'margin_left')
        row.prop(context.scene,'margin_right')
        
        
        row = layout.row()
        row = layout.row()
        row.operator(FSP_AddMargin.bl_idname, text="Add Margin to Pattern", icon='OBJECT_DATAMODE')



class FULLGRID_PT_export_mesh(FullGrid_panel, bpy.types.Panel):
    bl_label = "Export Current Pattern to Mesh"
    bl_parent_id = 'SD_PT_full_grid_main'
    
    def draw(self, context):
        layout = self.layout
        row = layout.row()  
        row.prop(context.scene, 'path_export_fullpattern')

       
        row = layout.row()
        row.prop(context.scene, 'filename_export_fullpattern')
        
        row = layout.row()
        row.prop(context.scene, 'export_format')
        
        row = layout.row()
        row = layout.row()
        row.operator(FSP_Export.bl_idname, text="Export", icon='EXPORT')

        
class SmockedGraph_panel(bpy.types.Panel):
    bl_label = "Smocked Graph"
    bl_idname = "SD_PT_smocked_graph"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SmockingDesign"
    bl_options ={"DEFAULT_CLOSED"}
   
    def draw(self, context):
        layout = self.layout
        
    
        
# ========================================================================
#                          Registration
# ========================================================================

wm = bpy.types.WindowManager
wm.usp_drawing_started = bpy.props.BoolProperty(default=False)
wm.fsp_drawing_started = bpy.props.BoolProperty(default=False)


_classes = [
    
    debug_clear,
    debug_print,
    debug_func,

    debug_panel,
    
    
    USP_CreateGrid,
    USP_SaveCurrentStitchingLine,
    USP_SelectStitchingPoint,
    USP_FinishCurrentDrawing,
    USP_FinishPattern,
    ExportUnitPattern,
    ImportUnitPattern,
        
    
    FSP_Tile,
    FSP_AddMargin,
    FSP_Export,
    FSP_DeleteStitchingLines_start,
    FSP_DeleteStitchingLines_done,
    FSP_EditMesh_start,
    FSP_EditMesh_done,
    FSP_AddStitchingLines_draw_start,
    FSP_AddStitchingLines_draw_end,
    FSP_AddStitchingLines_draw_add,    
    FSP_AddStitchingLines_draw_finish,
    FSP_CombinePatterns_load_first,
    FSP_CombinePatterns_load_second,
    FSP_CombinePatterns,
    
    
    UNITGRID_PT_main,
    UNITGRID_PT_create,
    UNITGRID_PT_load,
    
    FULLGRID_PT_main,
    FULLGRID_PT_tile,
    FULLGRID_PT_combine_patterns,
    FULLGRID_PT_edit_pattern,
    FULLGRID_PT_add_margin,
    FULLGRID_PT_export_mesh,
    
    SmockedGraph_panel
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