import numpy as np 
import binvox_rw
import pickle
def save_voxel(voxel_, filename, THRESHOLD=0.5):
  S1 = voxel_.shape[2]
  S2 = voxel_.shape[1]
  S3 = voxel_.shape[0]

  binvox_obj = binvox_rw.Voxels(
    np.transpose(voxel_, [1, 2, 0]) >= THRESHOLD,
    dims = [S1, S2, S3],
    translate = [0.0, 0.0, 0.0],
    scale = 1.0,
    axis_order = 'xyz'
  )   

  with open(filename, "wb") as f:
    binvox_obj.write(f)

angles = [(0,20)]
# val = pickle.load(open("827.p","rb"))
import pickle
blocks = pickle.load(open("val.p","rb"))
save_voxel(blocks,"transpose1.binvox")