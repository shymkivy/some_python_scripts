# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 14:13:27 2020

@author: rylab_dataPC
"""
import cv2
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
#import deepdish

import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params

#%%
logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"\
                    "[%(process)d] %(message)s",
                    level=logging.INFO) #, INFO, WARNING
 
#%%

f_dir = 'F:\\data\\Auditory\\caiman_out\\movies\\';
f_name = 'A1_ammn1_10_2_18_OA_cut_very_short';
f_ext = '.tif';

fnames = [f_dir + f_name + f_ext];

#%% MC parameters

"""
Parameters
"""

# motion correction parameters
pw_rigid = True       # flag to select rigid vs pw_rigid motion correction
# maximum allowed rigid shift in pixels
#max_shifts = [int(a/b) for a, b in zip(max_shift_um, dxy)]
max_shifts = (6, 6) # in p
# start a new patch for pw-rigid motion correction every x pixels
#strides = tuple([int(a/b) for a, b in zip(patch_motion_um, dxy)])
strides = (96, 96)
# overlap between pathes (size of patch in pixels: strides+overlaps)
overlaps = (32, 32)
# maximum deviation allowed for patch with respect to rigid shifts
max_deviation_rigid = 3


mc_dict = {
    'fnames': fnames,
    'pw_rigid': pw_rigid,
    'max_shifts': max_shifts,
    'strides': strides,
    'overlaps': overlaps,
    'max_deviation_rigid': max_deviation_rigid,
    'border_nan': 'copy',
    'use_cuda': False
}

opts = params.CNMFParams(params_dict=mc_dict)

#%%

if 'dview' in locals():
        cm.stop_server(dview=dview)
c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None,
                                      single_thread=False)

# %%% MOTION CORRECTION
# first we create a motion correction object with the specified parameters
mc = MotionCorrect(fnames,  dview=None, **opts.get_group('motion')) #dview=dview,  dview=None


#%%

mc.motion_correct(save_movie=True)


#%%
dview.terminate();

#%%

