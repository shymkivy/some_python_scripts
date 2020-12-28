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

load_file_name = 'A1_ammn_3plt_2plm2_12_27_20_mpl1_cutMC_d1_256_d2_256_d3_1_order_F_frames_27418_.mmap';

save_file_name = 'A1_ammn_3plt_2plm2_12_27_20_mpl1_cutMC.tif';

#%%
x3 = cm.load(f_dir_mov+load_file_name);

#%%

x3.save(f_dir_mov+save_file_name);