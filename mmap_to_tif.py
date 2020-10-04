# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 13:22:14 2020

@author: rylab_dataPC
"""


import numpy as np
import caiman as cm



#%%

f_dir_mov = 'F:\\data\\Auditory\\caiman_out\\\movies\\';

load_file_name = 'A1_ammn1_10_2_18_OA_cutMC2_d1_256_d2_256_d3_1_order_F_frames_24396_.mmap';

save_file_name = 'A1_ammn1_10_2_18_OA_cut_MC2_pw_rig.tif';

#%%
x3 = cm.load(f_dir_mov+load_file_name);

#%%

x3.save(f_dir_mov+save_file_name);