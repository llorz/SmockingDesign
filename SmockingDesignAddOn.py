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
from bpy.types import Operator
from bpy.types import (Panel, Operator)


from bpy.props import FloatVectorProperty
from bpy_extras.object_utils import AddObjectHelper, object_data_add
from mathutils import Vector


def delete_all():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    
def clean_objects() -> None:
    for item in bpy.data.objects:
        bpy.data.objects.remove(item)


def execute_operator(self, context):
    eval('bpy.ops.' + self.primitive + '()')
    
    
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

PROPS = [
    ('base_x', bpy.props.IntProperty(name='x', default=3, min=1, max=20)),
    ('base_y', bpy.props.IntProperty(name='y', default=3, min=1, max=20)),
    ('grid_display', bpy.props.EnumProperty(items=[("Solid","solid",""), ("Wire","wireframe","")], name='Display As',default="Wire",update=execute_operator)),
    ('add_version', bpy.props.BoolProperty(name='Add Version', default=False)),
    ('version', bpy.props.IntProperty(name='Version', default=1)),
]




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


#class addStitchingLines(Operator):
    

class restartPanel(bpy.types.Panel):
    bl_label = "Re"



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
        row.prop(context.scene,'grid_display')
        
        row = layout.row()
        row.operator(CreateGrid.bl_idname, text="Generate Grid", icon='GRID')


        
        layout.label(text= "Add Stitching Lines:")
        row = layout.row()
        row.operator(CreateGrid.bl_idname, text="Start", icon='LATTICE_DATA')
        row.operator(CreateGrid.bl_idname, text="Done", icon='CHECKMARK')

        layout.label(text= "Load Existing Patterns:")
        row = layout.row()
        
        

# Registration
_classes = [
    CreateGrid,
    GridPanel
 ]


def register():
    for (prop_name, prop_value) in PROPS:
        setattr(bpy.types.Scene, prop_name, prop_value)
        
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in _classes:
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
