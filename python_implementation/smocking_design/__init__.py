bl_info = {
    "name": "SmockingDesign",
    "author": "Jing Ren, Aviv Segall",
    "version": (1, 0),
    "blender": (2, 80, 0),
    "location": "View3D > N",
    "description": "Design and Simulate Smocking Arts",
    "warning": "",
    "doc_url": "",
    "category": "Design",
}

from smocking_design import SmockingDesignAddOn

def register():
    SmockingDesignAddOn.register()


def unregister():
    SmockingDesignAddOn.unregister()

if __name__ == "__main__":
    register()