import numpy as np

def get_underlay_nodes_ids(fsp, vids, sg):
  nl = fsp.num_stitching_lines()
  underlay_anchors = [vids[i] for i in range(len(vids)) if sg.dict_sp2sg[i] < nl]
  return underlay_anchors

def get_bbox(x):
  return np.amin(x, axis=0), np.amax(x, axis=0)

def center(x, bbox = None):
  if bbox is None:
    bbox = get_bbox(x)
  return x - (bbox[0] + bbox[1])/2

def rot_coord(x, y, r):
  theta = -x / r
  return np.array([r * np.cos(theta) - r * np.sin(theta),
  y,
  r * np.sin(theta) + r * np.cos(theta)])

def create_cylinder(flat_size, n = 40):
  r = flat_size[0] / (2 * np.pi)
  xx = np.linspace(0, flat_size[0], n)
  yy = np.linspace(0, flat_size[1], n)
  uv = np.array([[x, y] for x in xx for y in yy])
  v = np.array([[rot_coord(x, y, r)] for x,y in uv])
  return uv, v


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

  constraints = {underlay_anchors[vid[i]]: verts[i] for i in range(len(vid)) if dists[i][vid[i]] < 1e-1}
  return constraints
