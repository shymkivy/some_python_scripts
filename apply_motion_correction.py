# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 16:54:50 2020

@author: rylab_dataPC
"""
import glob
import numpy as np
import os
import logging
import matplotlib.pyplot as plt

import caiman as cm
from caiman.paths import caiman_datadir
from caiman.source_extraction import cnmf as cnmf
from caiman.utils.utils import download_demo
import h5py
import time


#%%

f_dir_data = 'F:\\data\\Auditory\\caiman_out\\OA_outputs\\';
f_name_data = 'A1_ammn1_10_2_18_OA_cut_ar2_5gsig_results.hdf5';

f_dir_mov = 'F:\\data\\Auditory\\caiman_out\\\movies\\';
f_name_mov = 'A1_ammn1_10_2_18_OA_cut';
f_ext_mov = '.hdf5';


#%%

cnm_load = cnmf.online_cnmf.load_OnlineCNMF(f_dir_data+f_name_data)
#%%

if 'dview' in locals():
        cm.stop_server(dview=dview)
c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None,
                                     single_thread=False)



if cnm_load.params.online['motion_correct']:
    shifts = cnm_load.estimates.shifts[-cnm_load.estimates.C.shape[-1]:]
    if not cnm_load.params.motion['pw_rigid']:
        images = cm.load(f_dir_mov+f_name_mov+f_ext_mov)
        memmap_file = cm.motion_correction.apply_shift_online(images, shifts,
                                            save_base_name=(f_dir_mov+f_name_mov+'MC'))
        
    else:
        mc = cm.motion_correction.MotionCorrect(f_dir_mov+f_name_mov+f_ext_mov, dview=dview,
                                            **cnm_load.params.get_group('motion'))
        
        mc.y_shifts_els = [[sx[1] for sx in sh] for sh in shifts]
        mc.x_shifts_els = [[sx[0] for sx in sh] for sh in shifts]

        memmap_file = mc.apply_shifts_movie(f_dir_mov+f_name_mov+f_ext_mov,
                                                save_memmap=True,
                                                save_base_name=(f_dir_mov+f_name_mov+'MC2'))

else:  # To do: apply non-rigid shifts on the fly
    images = cm.load(f_dir_mov+f_name_mov+f_ext_mov)
    memmap_file = images.save(f_dir_mov+f_name_mov+'.mmap')
    
    
dview.terminate();

#%%
plt.figure();
#%%
x1 = cnm_load.estimates.shifts
#%%
plt.figure;
plt.plot(x1[:,1,1])

#%%