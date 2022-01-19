#!/usr/bin/env python -W ignore::DeprecationWarning
# -*- coding: utf-8 -*-

"""
Complete pipeline for online processing using CaImAn Online (OnACID).
The demo demonstates the analysis of a sequence of files using the CaImAn online
algorithm. The steps include i) motion correction, ii) tracking current 
components, iii) detecting new components, iv) updating of spatial footprints.
The script demonstrates how to construct and use the params and online_cnmf
objects required for the analysis, and presents the various parameters that
can be passed as options. A plot of the processing time for the various steps
of the algorithm is also included.
@author: Eftychios Pnevmatikakis @epnev
Special thanks to Andreas Tolias and his lab at Baylor College of Medicine
for sharing the data used in this demo.
"""

import glob
import numpy as np
import os
import logging
import matplotlib.pyplot as plt

try:
    if __IPYTHON__:
        # this is used for debugging purposes only.
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass

import caiman as cm
from caiman.paths import caiman_datadir
from caiman.source_extraction import cnmf as cnmf
from caiman.utils.utils import download_demo
import h5py
import time


# %%
def main():
    pass # For compatibility between running under Spyder and the CLI

# %%  download and list all files to be processed
    #
    ## folder inside ./example_movies where files will be saved
    #fld_name = 'Mesoscope'
    #download_demo('Tolias_mesoscope_1.hdf5', fld_name)
    #download_demo('Tolias_mesoscope_2.hdf5', fld_name)
    #download_demo('Tolias_mesoscope_3.hdf5', fld_name)
    #
    ## folder where files are located
    #folder_name = os.path.join(caiman_datadir(), 'example_movies', fld_name)
    #extension = 'hdf5'                                  # extension of files
    ## read all files to be processed
    #fnames = glob.glob(folder_name + '/*' + extension)
    #
    ## your list of files should look something like this
    #logging.info(fnames)
    #
    #
      
    
    start_t = time.time()
    
    #f_dir = 'E:\data\V1\proc_data\\'
    #f_dir = 'E:\\data\\Auditory\\caiman_out_multiplane\\'
    #f_dir = 'G:\\data\\Auditory\\caiman_out\\movies\\'
    #f_dir = 'C:\\Users\\ys2605\\Desktop\\stuff\\AC_data\\cmnf_data\\'
    f_dir = 'C:\\Users\\ys2605\\Desktop\\stuff\\AC_data\\caiman_data\\movies\\'
    #f_dir = 'C:\\Users\\ys2605\\Desktop\\stuff\\random_save_path\\movies\\'
    #f_name = 'A2_freq_grating1_10_2_18_OA_cut'
    #f_dir = 'C:\\Users\\rylab_dataPC\\Desktop\\Yuriy\\DD_data\\proc_data\\'
    #f_name = 'vmmn2_9_16_19a_OA_cut'
    #f_name = 'ammn_2_dplanes2_10_14_19_OA_mpl1_cut';
    f_name = 'A1_cont_1_12_4_21b_mpl5_pl3'; # A1_cont_4_12_4_21b_mpl5_pl4
    f_ext = 'h5'
    fnames = [f_dir + f_name + '.' + f_ext]
    
    # %%   Set up some parameters
    
    fr = 10; #9.3273 for 5mpl; ~ 10  # frame rate (Hz) 3pl + 4ms = 15.5455; 55l+4 = 9.3273
    decay_time = 0.5 # 2 for s 0.5 for f # approximate length of transient event in seconds
    gSig = (6,6)  # expected half size of neurons
    p = 2  # order of AR indicator dynamics
    min_SNR = 1   # minimum SNR for accepting new components
    ds_factor = 1  # spatial downsampling factor (increases speed but may lose some fine structure)
    gnb = 2  # number of background components
    gSig = tuple(np.ceil(np.array(gSig) / ds_factor).astype('int')) # recompute gSig if downsampling is involved
    mot_corr = False  # flag for online motion correction
    pw_rigid = True  # flag for pw-rigid motion correction (slower but potentially more accurate)
    max_shifts_online = 6  # maximum allowed shift during motion correction
    sniper_mode = True  # use a CNN to detect new neurons (o/w space correlation)
    rval_thr = 0.9  # soace correlation threshold for candidate components
    # set up some additional supporting parameters needed for the algorithm
    # (these are default values but can change depending on dataset properties)
    init_batch = 500  # number of frames for initialization (presumably from the first file)
    K = 1  # initial number of components
    epochs = 2  # number of passes over the data
    show_movie = True # show the movie as the data gets processed
    merge_thr = 0.8
    
    params_dict = {'fnames': fnames,
                   'fr': fr,
                   'decay_time': decay_time,
                   'gSig': gSig,
                   'p': p,
                   'min_SNR': min_SNR,
                   'rval_thr': rval_thr,
                   'merge_thr': merge_thr,
                   'ds_factor': ds_factor,
                   'nb': gnb,
                   'motion_correct': mot_corr,
                   'init_batch': init_batch,
                   'init_method': 'bare',
                   'normalize': True,
                   'sniper_mode': sniper_mode,
                   'K': K,
                   'epochs': epochs,
                   'max_shifts_online': max_shifts_online,
                   'pw_rigid': pw_rigid,
                   'dist_shape_update': True,
                   'min_num_trial': 10,
                   'show_movie': show_movie}
    
    
    # 'path_to_model': 'C:\Users\ys2605\Anaconda3\envs\caiman2\share\caiman\model',
    
    opts = cnmf.params.CNMFParams(params_dict=params_dict)
    
    # %% fit online
    
    cnm = cnmf.online_cnmf.OnACID(params=opts)
    
    cnm.fit_online()
    
    # %% plot contours (this may take time)
    logging.info('Number of components: ' + str(cnm.estimates.A.shape[-1]))
    images = cm.load(fnames)
    Cn = images.local_correlations(swap_dim=False, frames_per_chunk=500)
    cnm.estimates.plot_contours(img=Cn, display_numbers=False)
    
    # %% view components
    cnm.estimates.view_components(img=Cn)
    
    # %% plot timing performance (if a movie is generated during processing, timing
    # will be severely over-estimated)
    
    T_motion = 1e3*np.array(cnm.t_motion)
    T_detect = 1e3*np.array(cnm.t_detect)
    T_shapes = 1e3*np.array(cnm.t_shapes)
    T_track = 1e3*np.array(cnm.t_online) - T_motion - T_detect - T_shapes
    plt.figure()
    plt.stackplot(np.arange(len(T_motion)), T_motion, T_track, T_detect, T_shapes)
    plt.legend(labels=['motion', 'tracking', 'detect', 'shapes'], loc=2)
    plt.title('Processing time allocation')
    plt.xlabel('Frame #')
    plt.ylabel('Processing time [ms]')
    #%% RUN IF YOU WANT TO VISUALIZE THE RESULTS (might take time)
    if 'dview' in locals():
        cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None,
                                     single_thread=False)
    
    mc_name_tag = 'MCPWrigid' + str(opts.motion['pw_rigid'])
    
    if opts.online['motion_correct']:
        shifts = cnm.estimates.shifts[-cnm.estimates.C.shape[-1]:]
        if not opts.motion['pw_rigid']:
            memmap_file = cm.motion_correction.apply_shift_online(images, shifts,
                                                        save_base_name=(f_dir+f_name+mc_name_tag))
        else:
            mc = cm.motion_correction.MotionCorrect(fnames, dview=dview,
                                                    **opts.get_group('motion'))
            
            mc.x_shifts_els = [[sx[0] for sx in sh] for sh in shifts]
            mc.y_shifts_els = [[sx[1] for sx in sh] for sh in shifts]
            
            memmap_file = mc.apply_shifts_movie(fnames, rigid_shifts=False,
                                                save_memmap=True,
                                                save_base_name=(f_dir+f_name+mc_name_tag))
    else:  # To do: apply non-rigid shifts on the fly
        memmap_file = images.save(fnames[0][:-3] + mc_name_tag + '.mmap')
        
        
    cnm.mmap_file = memmap_file
    Yr, dims, T = cm.load_memmap(memmap_file)
    
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    min_SNR = 2  # peak SNR for accepted components (if above this, acept)
    rval_thr = 0.85  # space correlation threshold (if above this, accept)
    use_cnn = True  # use the CNN classifier
    min_cnn_thr = 0.99  # if cnn classifier predicts below this value, reject
    cnn_lowest = 0.1  # neurons with cnn probability lower than this value are rejected
    
    cnm.params.set('quality',   {'min_SNR': min_SNR,
                                'rval_thr': rval_thr,
                                'use_cnn': use_cnn,
                                'min_cnn_thr': min_cnn_thr,
                                'cnn_lowest': cnn_lowest})
    
    cnm.estimates.evaluate_components(images, cnm.params, dview=dview)
    cnm.estimates.Cn = Cn
    cnm.save(os.path.splitext(fnames[0])[0]+'_results.hdf5')
    
    
    dview.terminate()
    
    duration = time.time() - start_t
    print(duration/60)

#%%
# This is to mask the differences between running this demo in Spyder
# versus from the CLI
if __name__ == "__main__":
    main()
