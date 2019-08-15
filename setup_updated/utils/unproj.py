from scipy.misc import imread
import ipdb
st = ipdb.set_trace
from scipy.misc import imsave
# from nets import unproject
import voxel
import numpy as np
# st()
import imageio
import tensorflow as tf
tf.enable_eager_execution()
import binvox_rw

H = 128


def save_voxel(voxel_, filename, THRESHOLD=0.5):
  S1 = voxel_.shape[2]
  S2 = voxel_.shape[1]
  S3 = voxel_.shape[0]

  binvox_obj = binvox_rw.Voxels(
    np.transpose(voxel_, [2, 1, 0]) >= THRESHOLD,
    dims = [S1, S2, S3],
    translate = [0.0, 0.0, 0.0],
    scale = 1.0,
    axis_order = 'xyz'
  )   

  with open(filename, "wb") as f:
    binvox_obj.write(f)



def unproject(inputs, resize = False):

    if resize:
        inputs = tf.image.resize(inputs, (const.S, const.S))
    size = int(inputs.shape[1])

    #now unproject, to get our starting point
    inputs = voxel.unproject_image(inputs)

    #in addition, add on a z-map, and a local bias
    #copied from components.py
    meshgridz = tf.range(size, dtype = tf.float32)
    meshgridz = tf.reshape(meshgridz, (1, size, 1, 1))
    meshgridz = tf.tile(meshgridz, (1, 1, size, size))
    meshgridz = tf.expand_dims(meshgridz, axis = 4) 
    meshgridz = (meshgridz + 0.5) / (size/2) - 1.0 #now (-1,1)
    # st()
    #get the rough outline
    # unprojected_mask = tf.expand_dims(inputs[:,:,:,:,0], 4)
    # unprojected_depth = tf.expand_dims(inputs[:,:,:,:,0], 4)
    unprojected_depth = (tf.expand_dims(inputs[:,:,:,:,0], 4) - voxel.RADIUS) * (1/voxel.SCENE_SIZE)
    # return unprojected_depth
    if H > 32:
        outline_thickness = 0.1
    else:
        outline_thickness = 0.2
    # depth shell
    outline = tf.cast(tf.logical_and(
        unprojected_depth <= meshgridz,
        unprojected_depth + outline_thickness > meshgridz
    ), tf.float32)
    # outline *= unprojected_mask
    if True:
        #return tf.expand_dims(inputs[:,:,:,:,0], 4) #this is the unprojected mask
        return outline,unprojected_depth

    inputs_ = [inputs]
    if const.USE_MESHGRID:
        inputs_.append(meshgridz)
    if const.USE_OUTLINE:
        inputs_.append(outline)
    inputs = tf.concat(inputs_, axis = 4)
    return inputs
# st()
# img = np.expand_dims(imread("img.png")[:,:,-1],-1)
# depth = np.expand_dims(imread("depth.png","L"),-1)
# depth = np.array(imageio.imread("rendered_depth.exr", format='EXR-FI'))[:,:,0]

def run(ricson,fov):
    if ricson:
        import pickle
        depths = pickle.load(open("pickled_ricson/depths.p","rb"))
        voxels = pickle.load(open("pickled_ricson/voxel.p","rb"))
        images = pickle.load(open("pickled_ricson/images.p","rb"))

        depth = depths[0][0,:,:,0]
        image = images[0][0,:,:,0]
        voxel_gt= voxels[0]
        imsave("depth_ric.png",depth)
        imsave("image_ric.png",image)
        voxel.RADIUS = 4.0
        voxel.SCENE_SIZE =1.0
        voxel.NEAR = voxel.RADIUS - voxel.SCENE_SIZE
        voxel.FAR = voxel.RADIUS + voxel.SCENE_SIZE
        voxel.fov =30.0
        voxel.W = 128
        # st()
        # depth =np.transpose(depth, [1, 0])
        # depth = depth * (100 - 0) + 0
        depth.astype(np.float32)
        # st()
        val = np.expand_dims(np.expand_dims(depth,axis=-1),0)
        val,unprojected_depth = unproject(val)
        # print(val.shape)
        # sess = tf.Session()
        # import pickle
        # st()
        unprojected_depth = np.squeeze(unprojected_depth)
        val = np.squeeze(val)
        save_voxel(val, "pickled_ricson/outline_ric.binvox")
        save_voxel(unprojected_depth, "pickled_ricson/unproj_depth_ric.binvox")
        save_voxel(voxel_gt, "pickled_ricson/ground_truth_ric.binvox")
    else:
        voxel.RADIUS = 12.0
        voxel.SCENE_SIZE =10.0
        voxel.NEAR = voxel.RADIUS - voxel.SCENE_SIZE
        voxel.FAR = voxel.RADIUS + voxel.SCENE_SIZE
        voxel.fov =fov
        voxel.W = 64.0
        depth = np.array(imageio.imread("ashar_data/output_gt.binvox", format='EXR-FI'))[:,:,0]
        # depth =np.transpose(depth, [1, 0])
        depth = depth * (100 - 0) + 0
        depth.astype(np.float32)
        # st()
        val = np.expand_dims(np.expand_dims(depth,axis=-1),0)
        val,unprojected_depth = unproject(val)
        # print(val.shape)
        # sess = tf.Session()
        # import pickle
        # st()
        unprojected_depth = np.squeeze(unprojected_depth)
        val = np.squeeze(val)
        save_voxel(val, "ashar_data_unproj/outline_{}.binvox".format(fov))
        save_voxel(unprojected_depth, "ashar_data_unproj/unproj_depths_{}.binvox".format(fov))

# [run(False,i) for i in [20,40,60,80,100]]
run(False,20)
