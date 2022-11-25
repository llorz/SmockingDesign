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



col_blue = (76/255.0, 201/255.0,240/255.0)
col_yellow = (254/255.0, 228/255.0, 64/255.0)




# ------------------------------------------------------------------------
#    global variables
# ------------------------------------------------------------------------

PROPS = [
    ('base_x', bpy.props.IntProperty(name='X', default=3, min=1, max=20)),
    ('base_y', bpy.props.IntProperty(name='Y', default=3, min=1, max=20)),
    ('num_x', bpy.props.FloatProperty(name='Xtile', default=3, min=1, max=20)),
    ('num_y', bpy.props.FloatProperty(name='Ytile', default=3, min=1, max=20)),
    ('shift_x', bpy.props.FloatProperty(name='Xshift', default=0, min=-1, max=1)),
    ('shift_y', bpy.props.FloatProperty(name='Yshift', default=0, min=-1, max=1)),
    ('type_tile', bpy.props.EnumProperty(items = [("regular", "regular", "tile in a regular grid"), ("radial", "radial", "tile in a radial grid")], name="Tiling Type", default="regular"))]
    

class StitchingLinesProp():
    currentDrawing = []
    # so we dont show multiple current drawings when repeated click "Done"
    if_curr_drawing_is_shown = False 
    
    if_user_is_drawing = True
    
    savedStitchingLines = []
    
    colSaved = col_blue
    
    colTmp = col_yellow
    
# ------------------------------------------------------------------------
#    global variables
# ------------------------------------------------------------------------
def delete_collection():    
    for coll in bpy.data.collections:
        bpy.data.collections.remove(coll)
    

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
        

def add_text_to_scene(body="test",
                      location=(0,0,0),
                      scale=(1,1,1),
                      obj_name="font_obj"):
                          
    font_curve = bpy.data.curves.new(type="FONT", name="Font Curve")
    font_curve.body = body
    font_obj = bpy.data.objects.new(name=obj_name, object_data=font_curve)
    font_obj.location = location
    font_obj.scale = scale
#    bpy.context.scene.collection.objects.link(font_obj)
    bpy.context.scene.collection.children['SmockingPattern'].objects.link(font_obj)



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



# Drawing with Gpencil:
# Reference: 
# https://gist.github.com/blender8r/4688b3f05640737236c856bc7df47bee
# https://www.youtube.com/watch?v=csQNmnc5xQg

def create_line_stroke_from_gpencil(name="GPencil", line_width=12, coll_name='SmockingPattern'):
    gpencil_data = bpy.data.grease_pencils.new(name)
    gpencil = bpy.data.objects.new(gpencil_data.name, gpencil_data)
#    bpy.context.collection.objects.link(gpencil)
    bpy.context.scene.collection.children[coll_name].objects.link(gpencil)
    gp_layer = gpencil_data.layers.new("lines")

    gp_frame = gp_layer.frames.new(bpy.context.scene.frame_current)

    gp_stroke = gp_frame.strokes.new()
    gp_stroke.line_width = line_width
    gp_stroke.start_cap_mode = 'ROUND'
    gp_stroke.end_cap_mode = 'ROUND'
    gp_stroke.use_cyclic = False

    return gpencil, gp_stroke
    

def draw_stitching_line(pts, col): 
    gpencil, gp_stroke = create_line_stroke_from_gpencil("all_stitching_line")
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


def draw_saved_stitching_lines(context):
    props = bpy.context.scene.sl_props
    print(props.savedStitchingLines)
    for i in range(len(props.savedStitchingLines)):
        vids = props.savedStitchingLines[i]
        obj = bpy.data.objects['Grid']
        pts = get_vtx_pos(obj, vids)
        draw_stitching_line(pts, props.colSaved)
        
        
def delete_all_gpencil_objects():
    for obj in bpy.data.objects:
        if obj.type == 'GPENCIL':
            bpy.data.objects.remove(obj)
            
            
def construct_object_from_mesh_to_scene(V, F, mesh_name, coll_name='SmockingPattern'):
    # input: V nv-by-2/3 array, F list of array
    
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
#    bpy.context.scene.collection.objects.link(object)
    bpy.context.scene.collection.children[coll_name].objects.link(object)
    

def sort_edge(edges):
    # return the unique edges
    e_sort = [np.array([min(e), max(e)]) for e in edges]
    e_unique = np.unique(e_sort, axis = 0)
    return e_unique


def create_grid(num_x, num_y):
    xx = range(num_x+1) # [0, ..., num_x]
    yy = range(num_y+1) # [0, ..., num_y]    
    gx, gy = np.meshgrid(xx, yy)

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


class jing_test(Operator):
    bl_idname = "object.func_test"
    bl_label = "function to test"
    
    def execute(self, context):
        delete_collection()
        # New Collection
        sp_coll = bpy.data.collections.new("SmockingPattern")

        # Add collection to scene collection
        bpy.context.scene.collection.children.link(sp_coll)
                
        return {'FINISHED'}
# ------------------------------------------------------------------------
#    Drawing the Unit Pattern using mouse
# ------------------------------------------------------------------------

from bpy.props import IntProperty, FloatProperty

class SelectStitchingPointOperator(Operator):
    """Move an object with the mouse, example"""
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

        
class finishDrawing(Operator):
    bl_idname = "edit.finish_drawing"
    bl_label = "Finish Drawing the stitching line"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        
        bpy.ops.object.mode_set(mode = 'OBJECT')
        
        props = bpy.context.scene.sl_props
        
        if not props.if_curr_drawing_is_shown:
            mesh = bpy.data.objects['Grid']
            pts = get_vtx_pos(mesh, props.currentDrawing)
            draw_stitching_line(pts, props.colTmp)
            props.if_curr_drawing_is_shown = True

        props.if_user_is_drawing = False

        return {'FINISHED'}


class saveCurrentStitchingLine(Operator):
    bl_idname = "edit.save_current_stitching_line"
    bl_label = "Add one stiching line"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        props = bpy.context.scene.sl_props
        if props.currentDrawing:
            props.savedStitchingLines.append(props.currentDrawing)
        props.currentDrawing = []
        
        print(props.savedStitchingLines)
        
        delete_all_gpencil_objects()
        
        draw_saved_stitching_lines(context)
        
        props.if_curr_drawing_is_shown = True
        
        return {'FINISHED'}



class deleteCurrentStitchingLine(Operator):
    bl_idname = "edit.delete_current_stitching_line"
    bl_label = "delete the current one stiching line"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        props = bpy.context.scene.sl_props
        props.currentDrawing = []
        
        delete_all_gpencil_objects()
        
        draw_saved_stitching_lines(context)
        
        props.if_curr_drawing_is_shown = True
        
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
        # there is some but in blender subdivison
#        scale = max(base_x, base_y)
#        # subdivison a bit buggy here
#        bpy.ops.mesh.primitive_grid_add(size=scale, 
#                                        location=(base_x/2, base_y/2,0),
#                                        x_subdivisions=base_x + 1,
#                                        y_subdivisions=base_y + 1) 
        
        
        # instead, manually create a grid
        gx, gy = create_grid(base_x, base_y)
        F, V, _ = extract_graph_from_meshgrid(gx, gy, True)
        construct_object_from_mesh_to_scene(V, F, 'Grid')
        
        
        
        mesh = bpy.data.objects['Grid']
        mesh.scale = (1, 1, 1)
        mesh.show_axis = False
        mesh.show_wire = True
        mesh.display_type = 'WIRE'
#        mesh.select_set(True) # select the grid
        
        add_text_to_scene(body="Unit Smocking Pattern", 
                          location=(0, base_y+0.5, 0), 
                          scale=(1,1,1),
                          obj_name='grid_annotation')
        
        for vid in range(len(mesh.data.vertices)):
            print(get_curr_vtx_pos(mesh, vid))
        
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
        base_x = context.scene.base_x
        base_y = context.scene.base_y
        
        num_x = context.scene.num_x
        num_y = context.scene.num_y
        shift_x = context.scene.shift_x
        shift_y = context.scene.shift_y
        
        props = bpy.context.scene.sl_props
        
        print(props.savedStitchingLines)
        
        print(num_x, num_y, shift_x, shift_y)
        return {'FINISHED'}
    
    
# ------------------------------------------------------------------------
#    define the add-on panel
# ------------------------------------------------------------------------


class GridPanel(bpy.types.Panel):
    bl_label = "Unit Smocking Pattern"
    bl_idname = "SD_PT_unit_grid"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SmockingDesign"
    
    def draw(self, context):
        
        layout = self.layout
        
        
        layout.label(text= "Create Unit Grid:")
        row = layout.row()
        row.prop(context.scene,'base_x')
        row.prop(context.scene,'base_y')

        
        row = layout.row()
        row.operator(CreateGrid.bl_idname, text="Generate Grid", icon='GRID')


        
        layout.label(text= "Draw A Stitching Line")
        row = layout.row()
        
        row.operator(SelectStitchingPointOperator.bl_idname, text="Draw", icon='LATTICE_DATA')
        row.operator(finishDrawing.bl_idname, text="Done", icon='CHECKMARK')
       
        layout.label(text= "Save Current Drawing...or not")
        row = layout.row()
        row.operator(saveCurrentStitchingLine.bl_idname, text="Add", icon='FUND')
        row.operator(deleteCurrentStitchingLine.bl_idname, text="Delete", icon='HEART')
        
               
        
        layout.label(text= "Load Existing Patterns:")
        row = layout.row()
        
class FullGridPanel(bpy.types.Panel):
    bl_label = "Full Smocking Pattern"
    bl_idname = "SD_PT_full_grid"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SmockingDesign"
    
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
        
        row = layout.row()
        row.operator(jing_test.bl_idname, text="test", icon='GHOST_ENABLED')
        


# ------------------------------------------------------------------------
#    Registration
# ------------------------------------------------------------------------

_classes = [
    CreateGrid,
    saveCurrentStitchingLine,
    deleteCurrentStitchingLine,
    SelectStitchingPointOperator,
    finishDrawing,
    
    jing_test,
    
    FullSmockingPattern,
    GridPanel,
    FullGridPanel
 ]


def register():
    for (prop_name, prop_value) in PROPS:
        setattr(bpy.types.Scene, prop_name, prop_value)
        
    for cls in _classes:
        bpy.utils.register_class(cls)
        
    bpy.types.Scene.sl_props = StitchingLinesProp()

def unregister():
    for cls in _classes:
        bpy.utils.unregister_class(cls)
        
    del bpy.types.Scene.sf_props    

if __name__ == "__main__":
    register()





# useless

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
