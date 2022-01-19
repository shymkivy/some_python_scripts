# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 13:22:14 2020

@author: rylab_dataPC
"""


import numpy as np
import caiman as cm



#%%

#f_dir_mov = 'F:\\data\\Auditory\\caiman_out\\\movies\\';
f_dir_mov = 'C:\\Users\\ys2605\\Desktop\\stuff\\AC_data\\caiman_data\\movies\\'


load_file_name = 'A1_cont_0.5_12_4_21a_mpl5_pl1_cut_bidi_moMCPWrigidTrue_d1_256_d2_256_d3_1_order_F_frames_19245_.mmap';

save_file_name = 'A1_cont_0.5_12_4_21a_mpl5_pl1_cut_bidi_moMCPWrigidTrue_d1_256_d2_256_d3_1_order_F_frames_19245_.tif';


#%%
x3 = cm.load(f_dir_mov+load_file_name);

#%%

x3.save(f_dir_mov+save_file_name);