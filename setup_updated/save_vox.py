import numpy as np
import binvox_rw
def save_voxel(voxel_, filename, THRESHOLD=0.1):
    S1 = voxel_.shape[2]
    S2 = voxel_.shape[1]
    S3 = voxel_.shape[0]
    # st()
    binvox_obj = binvox_rw.Voxels(
        # zyx to xyz
        np.transpose(voxel_, [2, 1, 0]) > THRESHOLD,
        dims = [S1, S2, S3],
        translate = [0.0, 0.0, 0.0],
        scale = 1.0,
        axis_order = 'xyz'
    )   

    with open(filename, "wb") as f:
        binvox_obj.write(f)
