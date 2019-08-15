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
from utils.tfutil import _bytes_feature
#from z import read_zmap, read_norm
from bv import read_bv,read_bv_schematic
from ipdb import set_trace as st
import pickle
import json

const.H = 64
const.W = 64
const.S = 64
#IN_DIR = '/home/ricson/data/ShapeNetCore.v1/all_chairs'
# IN_DIR = '/home/sajaved/projects/text2scene/3dProbNeuralProgNet/data/CLEVR/clevr-dataset-gen/output/CLEVR_64_GEN_TEST/images/train/'
# OUT_DIR = '/home/sajaved/projects/text2scene/3dProbNeuralProgNet/data/CLEVR/clevr-dataset-gen/output/CLEVR_64_GEN_TEST/tfrs'
IN_DIR = '../CLEVR_64_36_AFTER_MAYHEM/images/train/'
OUT_DIR = 'tfrs_mayhem_tree'

PHIS = list(range(20, 80, 20))
# PHIS = list([40])
THETAS = list(range(0, 360, 30))

# THETAS = list(range(0, 360, 90))

listcompletedir = lambda x: [join(x,y) for y in listdir(x)]
listonlydir = lambda x: list(filter(isdir, listcompletedir(x)))

def enum_obj_paths():
    good_paths = []
    for path in listonlydir(IN_DIR):
        stuff = listdir(path)
        good_paths.append(path)
    print('%d data found' % len(good_paths))
    return good_paths[:400]

def parse_seg(arr):
    #red and green the two mugs
    #blue is the background
    return arr[:,:,:2]

def traverseTree(tree,nodeList=[]):
    for i in range(0, tree.num_children):
        nodeList = traverseTree(tree.children[i],nodeList)
    nodeList.append(tree)
    return nodeList


dictionary =['brown','cylinder','cube','yellow','sphere','right','cyan','blue','gray','rubber','purple','metal','green','red','small','large']

def np_for_obj(obj_path_):
    view_path = obj_path_
    images = []
    angles = []
    folder_name = view_path.split('/')[-1]
    # st()
    for phi in PHIS:
        for theta in THETAS:
            img_name = folder_name + '_%d_%d.png' % (theta, phi)
            img_path = join(view_path, img_name)
            images.append(imread(img_path).astype(np.int64))
            angles.append(np.array([phi, theta]).astype(np.float32))

            # z_name = 'invZ_%d_%d.npy' % (theta, phi)
            # z_path = join(view_path, z_name)
            # zmaps.append(np.expand_dims(np.load(z_path), axis = 3))

            # seg_name = 'amodal_%d_%d.png' % (theta, phi)
            # seg_path = join(view_path, seg_name)
            # segs.append(parse_seg(imread(seg_path)))
    binvox_path = view_path.replace("images","voxels")+ '.schematic'
    scene_path = view_path.replace("images","scenes") + '.json'

    tree_path = view_path.replace("images","trees") + '.tree'
    tree = pickle.load(open(tree_path,"rb"))

    nodeList = traverseTree(tree,[])
    tree_word = [i.word for i in nodeList]
    # st()
    one_hot = np.zeros([len(dictionary)])
    
    for i in tree_word:
        if i in dictionary:
            one_hot[dictionary.index(i)] =1
    # st()
    binvox = read_bv_schematic(binvox_path,scene_path)
    # st()
    # obj1_path = join(obj_path_, 'obj1.binvox')
    # obj2_path = join(obj_path_, 'obj2.binvox')
    # obj1 = read_bv(obj1_path)
    # obj2 = read_bv(obj2_path)
    

    return [np.stack(images, axis=0),
            np.stack(angles, axis=0),
            one_hot,
            np.zeros([36, const.H, const.W, 1]),
            
            # np.stack(segs, axis=0),
            binvox,folder_name,tree_path]


def tf_for_obj(obj_np):
    # if zmap true
        # obj_np[2] = 1.0 / (obj_np[2] + 1E-9) #convert to depth
        # #this is pre-scaling, so clip with these bounds
        # obj_np[2] = np.clip(obj_np[2], 1.5, 2.5)
    # st()
    print(obj_np[0].shape,"shapes")
    assert obj_np[0].shape == (36, const.H, const.W, 4)
    assert obj_np[1].shape == (36, 2)
    print("dictionary shape ",obj_np[2].shape)
    # st()
    assert obj_np[2].shape[0] == (len(dictionary))
    # dummy zmaps
    assert obj_np[3].shape == (36, const.H, const.W, 1)
    # assert obj_np[3].shape == (3 * 18, const.H, const.W, 2) #always have two objects...
    assert obj_np[4].shape == (const.S, const.S, const.S)
    # assert obj_np[5].shape == (const.S, const.S, const.S)
    # assert obj_np[6].shape == (const.S, const.S, const.S)    
    
    for obj in obj_np:
        if isinstance(obj, np.ndarray):
            obj[np.isnan(obj)] = 0.0
            
    # convert everything to f32 except categories
    images, angles, tree_constraint, zmaps,vox = list(map(np.float32, obj_np[:5]))
    # st()
    example = tf.train.Example(features=tf.train.Features(feature={
        'images': _bytes_feature(images.tostring()),
        'angles': _bytes_feature(angles.tostring()),
        'tree_constraint': _bytes_feature(tree_constraint.tostring()),
        'zmaps': _bytes_feature(zmaps.tostring()),
        'voxel': _bytes_feature(vox.tostring()),
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
        # st()
        # print(jobs)
        # job(jobs[0])
        p.map(job, jobs, chunksize = 1)
    else:
        for x in enumerate(enum_obj_paths()):
            job(x)


if __name__ == '__main__':
    main(False)  # set false for debug
