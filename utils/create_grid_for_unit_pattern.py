import bpy

bpy.ops.mesh.primitive_grid_add(size=2, 
                                location=(0,0,0),
                                x_subdivisions=2,
                                y_subdivisions=2)