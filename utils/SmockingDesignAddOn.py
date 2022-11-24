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
import numpy as np
from bpy.types import Operator
from bpy.types import (Panel, Operator)






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


def add_text_to_scene(body="test",
                      location=(0,0,0),
                      scale=(1,1,1),
                      obj_name="font_obj"):
                          
    font_curve = bpy.data.curves.new(type="FONT", name="Font Curve")
    font_curve.body = body
    font_obj = bpy.data.objects.new(name=obj_name, object_data=font_curve)
    font_obj.location = location
    font_obj.scale = scale
    bpy.context.scene.collection.objects.link(font_obj)



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

def create_line_stroke_from_gpencil(name="GPencil", line_width=12):
    gpencil_data = bpy.data.grease_pencils.new(name)
    gpencil = bpy.data.objects.new(gpencil_data.name, gpencil_data)
    bpy.context.collection.objects.link(gpencil)
    gp_layer = gpencil_data.layers.new("lines")

    gp_frame = gp_layer.frames.new(bpy.context.scene.frame_current)

    gp_stroke = gp_frame.strokes.new()
    gp_stroke.line_width = line_width
    gp_stroke.start_cap_mode = 'ROUND'
    gp_stroke.end_cap_mode = 'ROUND'
    gp_stroke.use_cyclic = False

    return gpencil, gp_stroke
    

def test_draw(pts): 
    gpencil, gp_stroke = create_line_stroke_from_gpencil("test")
    gp_stroke.points.add(len(pts))

    for item, value in enumerate(pts):
        gp_stroke.points[item].co = value
        gp_stroke.points[item].pressure = 10
        
    mat = bpy.data.materials.new(name="Black")
    bpy.data.materials.create_gpencil_data(mat)
    gpencil.data.materials.append(mat)
    mat.grease_pencil.show_fill = False
#    mat.grease_pencil.fill_color = (1.0, 0.0, 1.0, 1.0)
    mat.grease_pencil.color = (76/255.0, 201/255.0,240/255.0, 1.0)

    gp_stroke.points[0].pressure = 2
#    gp_stroke.points[0].vertex_color = (1.0, 0.0, 0.0, 1.0)
    gp_stroke.points[-1].pressure = 2
#    gp_stroke.points[-1].vertex_color = (0.0, 1.0, 0.0, 1.0)



# ------------------------------------------------------------------------
#    global variables
# ------------------------------------------------------------------------


PROPS = [
    ('base_x', bpy.props.IntProperty(name='x', default=3, min=1, max=20)),
    ('base_y', bpy.props.IntProperty(name='y', default=3, min=1, max=20))]
    

class StitchingLinesProp():
    currentDrawing = []
    if_user_is_drawing = True
    savedStitchingLines = []    

# ------------------------------------------------------------------------
#    mouse event detection
# ------------------------------------------------------------------------

from bpy.props import IntProperty, FloatProperty

class SelectStitchingPointOperator(bpy.types.Operator):
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
                            print(props.currentDrawing)
        
    #        elif event.type in {'RIGHTMOUSE', 'MIDMOUSE'}:
    #            print('I am here')
    #            props.if_user_is_drawing = False
    #            props.currentDrawing = []
    #            return {'CANCELLED'}
        else:
            return{'CANCELLED'}
        
        return {'PASS_THROUGH'}
        

    def invoke(self, context, event):
        
        props = bpy.context.scene.sl_props
        props.if_user_is_drawing = True

        if props.if_user_is_drawing:
            props.currentDrawing = []
            bpy.ops.object.mode_set(mode = 'EDIT') 
            
            obj = bpy.data.objects['Grid']
            bm = bmesh.from_edit_mesh(obj.data)
            
            for v in bm.verts:
                v.select = False
            
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            self.report({'WARNING'}, "No active object, could not finish")
            return {'CANCELLED'}
        
        

# ------------------------------------------------------------------------
#    functions
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
        
        # subdivison a bit buggy here
        bpy.ops.mesh.primitive_grid_add(size=2, 
                                        location=(base_x, base_y,0),
                                        x_subdivisions=base_x + 1,
                                        y_subdivisions=base_y + 1) 
        mesh = bpy.data.objects['Grid']
        mesh.scale = (base_x, base_y, 1)
        mesh.show_axis = False
        mesh.show_wire = True
        mesh.display_type = 'WIRE'
        
        add_text_to_scene(body="Unit Smocking Pattern", 
                          location=(0, base_y*2+0.5, 0), 
                          scale=(1,1,1),
                          obj_name='grid_annotation')
        
        
        return {'FINISHED'}





# TODO: some bug here - cannot click start twice :/
class startAddStitchingLines(Operator):
    """initialization for adding stitching lines"""
    bl_idname = "edit.initialize_add_stitching_lines"
    bl_label = "Initialization for Adding Stitching Lines"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        
        props = bpy.context.scene.sl_props
        
        currentDrawing = [] # clear the previously selected stitching lines
        
        mesh = bpy.data.objects['Grid']
        select_one_object(mesh)

        bpy.ops.object.mode_set(mode = 'EDIT') 
        
        return {'FINISHED'}

#class draw

class finishDrawing(Operator):
    bl_idname = "edit.finish_drawing"
    bl_label = "Finish Drawing the stitching line"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        
        bpy.ops.object.mode_set(mode = 'OBJECT')
        
        props = bpy.context.scene.sl_props
        
        mesh = bpy.data.objects['Grid']
        pts = get_vtx_pos(mesh, props.currentDrawing)
        test_draw(pts)
        
        print(pts)
#        for i in range(len(pts)-1):
#            test_draw(pts[i:i+2])
    
        props.if_user_is_drawing = False

        print('stop now')
        print(props.if_user_is_drawing)
        return {'FINISHED'}


class addOneStitchingLine(Operator):
    bl_idname = "edit.add_one_stitching_line"
    bl_label = "Add one stiching line"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        obj = bpy.data.objects['Grid']
        bm = bmesh.from_edit_mesh(obj.data)
        
#        if mouse_Lclick.positive :
#            for v in bm.verts:
#                if v.select:
#                    print(v.index)
#        
        
        vid_selected = []
        
        for v in bm.verts:
            if v.select:
                vid_selected.append(v.index)
        
        # TODO: dunno how to raise error :(
        if len(vid_selected) < 2:
            print('Error: Invalid Selection - A valid stitching line contains at least two nodes')
        else:
            # add the valid stitching line to the list
            UserStitchingLines.append(vid_selected) 
            
            # deselect all the points
            for vid in vid_selected:
                bm.verts[vid].select = False
            
            
            bpy.ops.object.mode_set(mode = 'OBJECT')
            pts = get_vtx_pos(obj, vid_selected)
            print(vid_selected)
            test_draw(pts)
            
            
        return {'FINISHED'}




# ------------------------------------------------------------------------
#    define the add-on panel
# ------------------------------------------------------------------------


class GridPanel(bpy.types.Panel):
    bl_label = "Unit Smocking Pattern"
    bl_idname = "panel_grid"
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


        
        layout.label(text= "Add Stitching Lines:")
        row = layout.row()
        
        row.operator(SelectStitchingPointOperator.bl_idname, text="Draw a stitching line", icon='LATTICE_DATA')
        row.operator(finishDrawing.bl_idname, text="Done", icon='CHECKMARK')
        row = layout.row()
        row.operator(startAddStitchingLines.bl_idname, text="Start", icon='LATTICE_DATA')
        row = layout.row()
        row.operator(addOneStitchingLine.bl_idname, text="Add", icon='ADD')
               
        
        layout.label(text= "Load Existing Patterns:")
        row = layout.row()
        
        

# ------------------------------------------------------------------------
#    Registration
# ------------------------------------------------------------------------

_classes = [
    CreateGrid,
    startAddStitchingLines,
    addOneStitchingLine,
    SelectStitchingPointOperator,
    finishDrawing,
    GridPanel
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
