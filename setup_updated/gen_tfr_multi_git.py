#!/usr/bin/env python2
#/home/sajaved/projects/3dmapping/setup_updated
import sys
sys.path.append('..')
import constants as const
from scipy.misc import imread
import numpy as np
import tensorflow as tf
from os import listdir
from os.path import join, isdir
import pathos.pools as pp
from utils.tfutil import _bytes_feature, _int64_feature, _int64s_feature
#from z import read_zmap, read_norm
from bv import read_bv,read_bv_schematic_multi,read_bv_schematic
from ipdb import set_trace as st
import pickle
import imageio
import os.path
import json

const.H = 64
const.W = 64
const.S = 64
#IN_DIR = '/home/ricson/data/ShapeNetCore.v1/all_chairs'
# IN_DIR = '/home/sajaved/projects/text2scene/3dProbNeuralProgNet/data/CLEVR/clevr-dataset-gen/output/CLEVR_64_GEN_TEST/images/train/'
# OUT_DIR = '/home/sajaved/projects/text2scene/3dProbNeuralProgNet/data/CLEVR/clevr-dataset-gen/output/CLEVR_64_GEN_TEST/tfrs'

root_dir = "../"
IN_DIR = os.path.join(root_dir, 'CLEVR_64_36_MORE_OBJ_FINAL/images/train/')
OUT_DIR = 'tfrs_names'

#IN_DIR = '../CLEVR_64_36_TRIAL/images/train/'
#OUT_DIR = 'tfrs_rpn_2Obj'

PHIS = list(range(20, 80, 20))
# PHIS = list([40])
THETAS = list(range(0, 360, 30))

# THETAS = list(range(0, 360, 90))

listcompletedir = lambda x: [join(x,y) for y in listdir(x)]
listonlydir = lambda x: list(filter(isdir, listcompletedir(x)))

pad = 1

def bbox(object):
    object_idx = np.where(object==1)
    x_top = object_idx[0].min()
    y_top = object_idx[1].min()
    z_top = object_idx[2].min()

    x_bottom = object_idx[0].max()
    y_bottom = object_idx[1].max()
    z_bottom = object_idx[2].max()
    bbox = (x_top, y_top, z_top, x_bottom - x_top,
           y_bottom - y_top, z_bottom - z_top)
    return bbox
def enum_obj_paths():
    good_paths = []
    for path in listonlydir(IN_DIR):
        stuff = listdir(path)
        good_paths.append(path)
    print('%d data found' % len(good_paths))
    return good_paths

def parse_seg(arr):
    #red and green the two mugs
    #blue is the background
    return arr[:,:,:2]


def np_for_obj(obj_path_):
    view_path = obj_path_
    images = []
    angles = []
    depths = []
    masks = []
    folder_name = view_path.split('/')[-1]
    # st()
    scene_path = view_path.replace("images","scenes") + '.json'
    scene = json.load(open(scene_path))
    for phi in PHIS:
        for theta in THETAS:
            img_name = folder_name + '_%d_%d.png' % (theta, phi)
            img_path = join(view_path, img_name)
            depth_path  = img_path.replace("images","depth").replace("png","exr")
            images.append(imread(img_path).astype(np.float32))
            depth = np.array(imageio.imread(depth_path, format='EXR-FI'))[:,:,0]
            depth = depth * (100 - 0) + 0
            depth.astype(np.float32)
            depths.append(depth)
            angles.append(np.array([phi, theta]).astype(np.float32))
            mask = np.zeros([const.H,const.W,1])
            tp = str(theta) + "_" + str(phi)
            for obj in scene['objects']:
                ltop = obj["bbox_2d"][tp]["pixel_coords_lefttop"]
                rbot = obj["bbox_2d"][tp]["pixel_coords_rightbottom"]
                mask[ltop[1]-pad:rbot[1]+pad,ltop[0]-pad:rbot[0]+pad] = 1
            masks.append(mask)

            # z_name = 'invZ_%d_%d.npy' % (theta, phi)
            # z_path = join(view_path, z_name)
            # zmaps.append(np.expand_dims(np.load(z_path), axis = 3))

            # seg_name = 'amodal_%d_%d.png' % (theta, phi)
            # seg_path = join(view_path, seg_name)
            # segs.append(parse_seg(imread(seg_path)))
    # st()
    binvox_path = view_path.replace("images","voxels")+ '.schematic'
    scene_path = view_path.replace("images","scenes") + '.json'
    tree_path = view_path.replace("images","trees") + '.tree'
    # tree_path  = tree_path.split("/")[1:]
    tree_path = "/".join(tree_path.split("/")[2:])
    binvox = read_bv_schematic_multi(binvox_path,scene_path)
    print(np.unique(binvox))
    # st()
    return [np.stack(images, axis=0),
            np.stack(angles, axis=0),
            np.expand_dims(np.stack(depths, axis=0), axis = 3),
            np.stack(masks, axis=0),
            # np.stack(segs, axis=0),
            binvox,folder_name,tree_path]


def tf_for_obj(obj_np):
    assert obj_np[0].shape == (36, const.H, const.W, 4)
    assert obj_np[1].shape == (36, 2)
    # dummy zmaps
    assert obj_np[2].shape == (36, const.H, const.W, 1)
 
    val = {"0":[0,0,0,0],"1":[1,0,0,0],"2":[1,1,0,0],"3":[1,1,1,0]}
    for obj in obj_np:
        if isinstance(obj, np.ndarray):
            obj[np.isnan(obj)] = 0.0
    voxel_obj= np.zeros([4,const.S,const.S,const.S],np.float32)
    num_objs = len(obj_np[4])
    print("num objs ",num_objs)
    # if num_objs!=2:
    #     raise Exception("more objects than 1")

    voxel_obj[:num_objs] = obj_np[4]

    # convert everything to f32 except categories
    images, angles, zmaps,mask= list(map(np.float32, obj_np[:4]))

    example = tf.train.Example(features=tf.train.Features(feature={
        'images': _bytes_feature(images.tostring()),
        'depth': _bytes_feature(zmaps.tostring()),
        'mask': _bytes_feature(mask.tostring()),
        # 'voxel': _bytes_feature(voxel.tostring()),
        "voxel_obj" : _bytes_feature(voxel_obj.tostring()),
        'objs_mask': _bytes_feature(np.array(val[str(num_objs)],np.float64).tostring()),
        'file_name': _bytes_feature(obj_np[5].encode()),
        'treefile_path': _bytes_feature(obj_np[6].encode())
    }))
    return example


def out_path_for_obj_path(obj_path):
    return join(OUT_DIR, obj_path.split('/')[-1])


def write_tf(tfexample, path):
    compress = tf.io.TFRecordOptions(
        compression_type=tf.io.TFRecordCompressionType.GZIP)
    writer = tf.io.TFRecordWriter(path, options=compress)
    writer.write(tfexample.SerializeToString())
    writer.close()


def job(xxx_todo_changeme):
    (i, obj_path) = xxx_todo_changeme
    print(i, obj_path)
    # st()      
    out_path = out_path_for_obj_path(obj_path)
    tfexample = tf_for_obj(np_for_obj(obj_path))
    # st()
    write_tf(tfexample, out_path)



def main(mt):
    if mt:
        p = pp.ProcessPool(4)
        jobs = sorted(list(enumerate(enum_obj_paths())))

        p.map(job, jobs, chunksize = 1)
    else:
        for x in enumerate(enum_obj_paths()):
            job(x)


if __name__ == '__main__':
    main(True)  # set false for debug