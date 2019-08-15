import voxel
import pickle
import numpy as np
import ipdb
import tensorflow as tf
tf.enable_eager_execution()

st = ipdb.set_trace
# sess = tf.Session()
# st()
from nbtschematic import SchematicFile
import numpy as np
from scipy.misc import imsave
import binvox_rw
import pickle
# val = pickle.load(open("val_ricson.p","rb"))[0]
# pickle.dump(np.squeeze(val),open("val_ric1.p","wb"))
fov = [40.0]
#NO SCALING
import binvox_rw


# pickle.dump(blocks,open("827.p","wb"))
def save_voxel(voxel_, filename, THRESHOLD=0.5):
	S1 = voxel_.shape[2]
	S2 = voxel_.shape[1]
	S3 = voxel_.shape[0]
	# st()
	binvox_obj = binvox_rw.Voxels(
		np.transpose(voxel_, [2, 1, 0]) > THRESHOLD,
		dims = [S1, S2, S3],
		translate = [0.0, 0.0, 0.0],
		scale = 1.0,
		axis_order = 'xyz'
	)   

	with open(filename, "wb") as f:
		binvox_obj.write(f)


a =binvox_rw.read_as_3d_array(open("ashar_data/output_gt.binvox","rb"))
val = a.data
num = np.where(val)[0]
print("total",len(num))

val = tf.convert_to_tensor(np.expand_dims(np.expand_dims(val,0),-1))

fovs_working ={}
def genImage(angle,fov):
	global val
	theta,phi = angle
	proj_val = voxel.project_voxel(val,theta,phi,fov)
	num = np.where(proj_val>0.5)[0]
	if len(num) > 0:
		print("found")
		fovs_working[fov] = len(num)


	save_voxel(np.squeeze(proj_val),"ashar_data_rotate/valRotate_THETA_{}_PHI_{}.binvox".format(theta,phi))


# [genImage((0,20),i) for i in range(5, 100)]
genImage((0,0),70)
genImage((30,0),70)
genImage((60,0),70)
genImage((90,0),70)
print(fovs_working)
