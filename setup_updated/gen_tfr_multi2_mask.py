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
from scipy.misc import imsave
import sys
import json 

const.H = 64
const.W = 64
const.S = 64
#IN_DIR = '/home/ricson/data/ShapeNetCore.v1/all_chairs'
# IN_DIR = '/home/sajaved/projects/text2scene/3dProbNeuralProgNet/data/CLEVR/clevr-dataset-gen/output/CLEVR_64_GEN_TEST/images/train/'
# OUT_DIR = '/home/sajaved/projects/text2scene/3dProbNeuralProgNet/data/CLEVR/clevr-dataset-gen/output/CLEVR_64_GEN_TEST/tfrs'
IN_DIR = '/home/mprabhud/ashar_code/dataset/CLEVR_64_36_AFTER_CORRECTION_NO_DEPTH/images/train/'
OUT_DIR = '/home/mprabhud/ashar_code/dataset/CLEVR_64_36_AFTER_CORRECTION_NO_DEPTH/tfrs_withnames_voxel/'

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

def postorder_traversal(tree, node_list):
    if tree is None:
        return node_list
    if len(tree.children) > 0:
        node_list = postorder_traversal(tree.children[0], node_list)
    #node_list.append(tree)
    if len(tree.children) > 1:
        node_list = postorder_traversal(tree.children[1], node_list)
    node_list.append(tree)
    return node_list


def convert_to_list(tree):
    treex = []
    node_list = []
    node_list = postorder_traversal(tree, node_list)
    #st()
    for i, node in enumerate(node_list):
        tree_dict = {}
        tree_dict['word'] = node.word
        tree_dict['function'] = node.function
        children = [node_list.index(node.children[i]) for i in range(len(node.children))]
        if not children: children = [-1,-1]
        if len(children)==1: children = children + [-1]    
        tree_dict['children'] = children

        tree_dict['bbox'] = [-1]*6 if not hasattr(node, 'bbox') else node.bbox.tolist()
        print(tree_dict)
        treex.append(tree_dict)
    return treex

dictionary =['brown','cylinder','cube','yellow','sphere','right','cyan','blue','gray','rubber','purple','metal','green','red','small','large']

def np_for_obj(obj_path_):
    view_path = obj_path_
    images = []
    angles = []
    masks = []
    folder_name = view_path.split('/')[-1]
    # st()
    pad =1
    scene_path = view_path.replace("images","scenes") + '.json'
    scene = json.load(open(scene_path))
    for phi in PHIS:
        for theta in THETAS:
            img_name = folder_name + '_%d_%d.png' % (theta, phi)
            img_path = join(view_path, img_name)
            img_arr = imread(img_path).astype(np.int64)
            images.append(img_arr)
            angles.append(np.array([phi, theta]).astype(np.float32))
            mask = np.zeros([const.H,const.W,1])
            tp = str(theta) +"_" + str(phi)
            for obj in scene['objects']:
                ltop = obj["bbox_2d"][tp]["pixel_coords_lefttop"]
                rbot = obj["bbox_2d"][tp]["pixel_coords_rightbottom"]
                mask[ltop[1]-pad:rbot[1]+pad,ltop[0]-pad:rbot[0]+pad] = 1
            masks.append(mask)
            # imsave("mask_imgs"+"/"+'_%d_%d.png' % (theta, phi),mask*img_arr)


            # z_name = 'invZ_%d_%d.npy' % (theta, phi)
            # z_path = join(view_path, z_name)
            # zmaps.append(np.expand_dims(np.load(z_path), axis = 3))

            # seg_name = 'amodal_%d_%d.png' % (theta, phi)
            # seg_path = join(view_path, seg_name)
            # segs.append(parse_seg(imread(seg_path)))
    binvox_path = view_path.replace("images","voxels")+ '.schematic'

    
    
    # st()
    tree_path = view_path.replace("images","trees") + '.tree'
    tree = pickle.load(open(tree_path,"rb"))
    treeL = convert_to_list(tree)
    treeL_string = json.dumps(treeL)


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
            np.stack(masks, axis=0),
            
            
            # np.stack(segs, axis=0),
            binvox,folder_name,tree_path,treeL_string]


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
    images, angles, tree_constraint, mask,vox = list(map(np.float32, obj_np[:5]))
    # st()
    example = tf.train.Example(features=tf.train.Features(feature={
        'images': _bytes_feature(images.tostring()),
        'angles': _bytes_feature(angles.tostring()),
        'tree_constraint': _bytes_feature(tree_constraint.tostring()),
        'mask': _bytes_feature(mask.tostring()),
        'voxel': _bytes_feature(vox.tostring()),
        'file_name': _bytes_feature(obj_np[5].encode()),
        'treefile_path': _bytes_feature(obj_np[6].encode()),
        'tree_json': _bytes_feature(obj_np[7].encode())
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
