import numpy as np
import utils
from nbtschematic import SchematicFile
from ipdb import set_trace as st
import json

def read_bv(fn):
	with open(fn, 'rb') as f:
		model = utils.binvox_rw.read_as_3d_array(f)
	data = np.float32(model.data)
	return data

def read_bv_schematic(fn,scene_file):
	fileformat = 'schematic'
	with open(scene_file) as f:
		scene_json = json.load(f)
	scene_obj_list = scene_json['objects']
	print("num objs ",len(scene_obj_list))
	# check if extra object voxels need to be removed from blocks
	remove_extra_objects = True
	orig_block_id = int(scene_obj_list[0]['obj_id'].split('blockid_')[-1])
	if len(scene_obj_list) == 1:
		remove_extra_objects = True
		orig_block_id = int(scene_obj_list[0]['obj_id'].split('blockid_')[-1])
	# load image
	if fileformat == 'schematic':
		voxel_file = fn
		sf = SchematicFile.load(voxel_file)
		blocks = np.frombuffer(sf.blocks, dtype=sf.blocks.dtype)
		voxel_size = int(round(len(blocks)**(1./3)))
		blocks = blocks.reshape((voxel_size,voxel_size,voxel_size))
		blocks = blocks.copy()
		if remove_extra_objects:
			blocks[blocks != orig_block_id] = 0
	# st()
	data = np.float32(blocks)
	return data

def read_bv_schematic_multi(fn,scene_file):
	fileformat = 'schematic'
	with open(scene_file) as f:
		scene_json = json.load(f)
	scene_obj_list = scene_json['objects']

	# check if extra object voxels need to be removed from blocks
	remove_extra_objects = True
	main_blocks = None
	if fileformat == 'schematic':
		voxel_file = fn
		sf = SchematicFile.load(voxel_file)
		main_blocks = np.frombuffer(sf.blocks, dtype=sf.blocks.dtype)
		voxel_size = int(round(len(main_blocks)**(1./3)))
		main_blocks = main_blocks.reshape((voxel_size,voxel_size,voxel_size))
		main_blocks = main_blocks.transpose(1,2,0)

	objs =[]


	for i in range(len(scene_obj_list)):
		obj_id =int(scene_obj_list[i]['obj_id'].split('blockid_')[-1])
		blocks = main_blocks.copy()
		if remove_extra_objects:
			blocks[blocks != obj_id] = 0
		print(obj_id,"obj id")

		objs.append(np.float32(blocks//obj_id))

	return objs
