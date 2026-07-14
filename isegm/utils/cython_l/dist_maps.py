import pyximport
pyximport.install(pyximport=True, language_level=3)
# # noinspection PyUnresolvedReferences
from ._get_dist_maps import get_dist_maps

# import numpy as np
# import math

# def get_dist_maps(points, size):
#     h, w = size
#     n_points = points.shape[0]
#     dist_map = np.zeros((h, w), dtype=np.float32)

#     for y in range(h):
#         for x in range(w):
#             for i in range(n_points):
#                 dist_map[y, x] += math.sqrt((y - points[i, 0]) ** 2 + (x - points[i, 1]) ** 2)
    
#     return dist_map
