from mathutils import Vector
from mathutils.noise import random, seed_set
import bpy

# Specify parameters
seed = 0
xmin = -1
xmax = 1
ymin = -1
ymax = 1
radius = 0.1
num_circles = 68
max_tries = 10000

# Init
seed_set(seed)
sx = xmax-xmin-2*radius
sy = ymax-ymin-2*radius
xminm = xmin+radius 
yminm = ymin+radius 
existing_locations = []
sce = bpy.context.scene
bpy.ops.mesh.primitive_circle_add(location=(0,0,0), radius=radius)
ref_circle = bpy.context.object

# Loop
for i in range(num_circles):
    j = 0
    searchOn = True
    while searchOn:
        if j > max_tries:
            bpy.ops.object.select_all(action='DESELECT')
            ref_circle.select = True
            bpy.ops.object.delete() 
            raise ValueError('Found no more room for another circle')
            break
        j += 1
        new_location = (xminm + random()*sx,
                        yminm + random()*sy,
                        0)
        for existing_location in existing_locations:
            if (Vector(existing_location)-Vector(new_location)).length < 2*radius:
                break
        else:
            new_circle = ref_circle.copy()
            new_circle.location = new_location
            sce.objects.link(new_circle)
            existing_locations.append(new_location)
            searchOn = False

ref_circle.select = True
bpy.ops.object.delete() 
sce.update()