import numpy as np

def generate_hex_grid(numx, numy, R = 1, eps = 1e-6, cond = lambda x: True):
  h = np.sqrt(3) * R / 2
  no = 1 + (numx * 3 + 1) // 2
  ne = no - (numx % 2)
  # Bottom and top vertices of the "even" hexagons.
  Ve = [[x * R, y * 2 * h] for y in range(numy + 1) for x in range(ne)]
  # Middle of the "odd" hexagons.
  Vo = np.array([-0.5, h]) + [[x * R, y * 2 * h] for y in range(numy) for x in range(no) ]
  V = np.concatenate([Ve, Vo])
  E = [[vi, vj]
        for vi in range(V.shape[0]) for vj in range(V.shape[0]) 
        if abs(np.linalg.norm(V[vi] - V[vj]) - R) < eps]
  
  centers_Ve = [y*ne + x for y in range(numy + 1) for x in range(ne) if x % 3 == 2]
  centers_Vo =  [y * no + x + len(Ve) for y in range(numy) for x in range(no) if x % 3 == 1]
  # centers_Ve = np.array(range(2, len(Ve), 3))
  # centers_Vo = np.array(range(1, len(Vo), 3)) + len(Ve)
  centers = np.concatenate([centers_Ve, centers_Vo])
  return V, E, centers

def generate_radial_hex_grid(bigR, R = 1, eps = 1e-6):
  numx = bigR * 2 + 1
  numy = numx + bigR % 2
  # return generate_hex_grid(numx, numy, R, eps, lambda x,y:)
  