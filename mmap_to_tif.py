# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 13:22:14 2020

@author: rylab_dataPC
"""


import numpy as np
import caiman as cm

# import tifffle to save as tiff if caiman dont work

#%%

#f_dir_mov = 'F:\\data\\Auditory\\caiman_out\\\movies\\';
f_dir_mov = 'C:\\Users\\ys2605\\Desktop\\'


load_file_name = 'ch2_m1_NA05_1064nm150mW_fovx25_z130_sp_00001_d1_512_d2_512_d3_1_order_F_frames_21700_.mmap';

save_file_name = 'some_vid2_full.h5';


#%%
x3 = cm.load(f_dir_mov+load_file_name);

#%%

x3[:,:,:].save(f_dir_mov+save_file_name);