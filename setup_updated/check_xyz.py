from nbtschematic import SchematicFile
from  save_vox import save_voxel
import numpy as np


voxel_file = "../tmp/CLEVR_new_000735.schematic"
sf = SchematicFile.load(voxel_file)
main_blocks = np.frombuffer(sf.blocks, dtype=sf.blocks.dtype)
voxel_size = int(round(len(main_blocks)**(1./3)))
main_blocks = main_blocks.reshape((voxel_size,voxel_size,voxel_size))
main_blocks = main_blocks.transpose(1,2,0)
np.unique(main_blocks)

save_voxel(main_blocks,"check.binvox")