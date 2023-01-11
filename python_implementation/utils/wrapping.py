import numpy as np
import importlib
import itertools

import cpp_smocking_solver
importlib.reload(cpp_smocking_solver)

def get_underlay_nodes_ids(fsp, vids, sg):
    nl = fsp.num_stitching_lines()
    underlay_anchors = [vids[i]
                        for i in range(len(vids)) if sg.dict_sp2sg[i] < nl]
    return underlay_anchors


def get_bbox(x):
    return np.amin(x, axis=0), np.amax(x, axis=0)


def center(x, bbox=None):
    if bbox is None:
        bbox = get_bbox(x)
    return x - (bbox[0] + bbox[1])/2


def rot_coord(x, y, r):
    theta = -x / r
    return np.array([r * np.cos(theta) - r * np.sin(theta),
                     y,
                     r * np.sin(theta) + r * np.cos(theta)])

def extract_graph_from_meshgrid(gx, gy):
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
    
    F = np.array(list(itertools.chain.from_iterable(\
      [(f[[0,1,2]], f[[2,3,0]]) for f in F])))

    return F, V

def create_cylinder(flat_size, n=40):
    r = flat_size[0] / (2 * np.pi)
    xx = np.linspace(0, flat_size[0], n)
    yy = np.linspace(0, flat_size[1], n)
    gx, gy = np.meshgrid(xx, yy)
    F, uv = extract_graph_from_meshgrid(gx, gy)
    # uv = np.array([[x, y] for x in xx for y in yy])
    v = np.array([[rot_coord(x, y, r)] for x, y in uv])
    return uv, v, F


def get_constraints_from_param(underlay_locations, underlay_anchors, uv, verts):
    underlay_locations = underlay_locations[:, 0:2]
    # Get bbox of underlay.
    underlay_bbox = get_bbox(underlay_locations)
    underlay_locations -= (underlay_bbox[0] + underlay_bbox[1]) / 2
    # and uv.
    uv_bbox = get_bbox(uv)
    uv -= (uv_bbox[0] + uv_bbox[1])/2

    # Resize uv to fit in underlay_bbox
    ratio = (underlay_bbox[1] - underlay_bbox[0]) / (uv_bbox[1] - uv_bbox[0])
    min_ratio = min(ratio)
    # Only scale uv down if necessary.
    uv *= min(1., min_ratio)
    verts = center(verts) * min(1., min_ratio)

    # Find the underlay nodes that should be constrained.
    dists = [np.linalg.norm(underlay_locations - p, axis=1) for p in uv]
    vid = [np.argmin(x) for x in dists]

    constraints = {underlay_anchors[vid[i]]: verts[i]
                   for i in range(len(vid)) if dists[i][vid[i]] < 1e-1}
    return constraints


def get_constraints_from_param_bary(x, vids, fsp, sg, uv, verts, faces):
    underlay_anchors = get_underlay_nodes_ids(fsp, vids, sg)
    underlay_locations = x[underlay_anchors]
    underlay_locations = underlay_locations[:, 0:2]
    # Get bbox of underlay.
    underlay_bbox = get_bbox(underlay_locations)
    underlay_locations -= (underlay_bbox[0] + underlay_bbox[1]) / 2
    x -= np.concatenate([(underlay_bbox[0] + underlay_bbox[1]) / 2, [0]])
    # and uv.
    uv_bbox = get_bbox(uv)
    uv -= (uv_bbox[0] + uv_bbox[1])/2

    # Resize uv to fit in underlay_bbox
    ratio = (underlay_bbox[1] - underlay_bbox[0]) / (uv_bbox[1] - uv_bbox[0])
    min_ratio = min(ratio)
    # Only scale uv down if necessary.
    uv *= min(1., min_ratio)
    verts = center(verts) * min(1., min_ratio)

    # Find the underlay nodes that should be constrained.
    coords = cpp_smocking_solver.bary_coords(x, uv, faces)
    delete_verts = []
    constraints = {}
    constraint_weight = {}
    for i in range(len(x)):
        # Pleat/underlay outside of the parameterization domain.
        f, l1, l2, l3 = coords[i]
        face = faces[int(f)]
        if (f < 0):
            delete_verts.append(i)
            continue
        # Pleat that should not be removed but doesn't have any constraints.
        # if i not in vids:
          # continue
        if i not in underlay_anchors:
            v1, v2, v3 = verts[face[0]], verts[face[1]], verts[face[2]]
            mesh_pos = l1 * v1 + l2 * v2 + l3 * v3
            n = np.cross(v2-v1, v3-v2)
            n = n / np.linalg.norm(n)
            constraint_weight[i] = 0.01
            constraints[i] = mesh_pos + n * x[i, 2]
            continue
        # Get constraint from the location on the triangle.
        constraint_weight[i] = 1
        constraints[i] = l1 * verts[faces[int(f)][0]] + l2 * \
            verts[faces[int(f)][1]] + l3 * verts[faces[int(f)][2]]

    return constraints, constraint_weight, delete_verts
