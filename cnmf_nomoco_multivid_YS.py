#!/usr/bin/env python

"""
Complete demo pipeline for processing two photon calcium imaging data using the
CaImAn batch algorithm. The processing pipeline included motion correction,
source extraction and deconvolution. The demo shows how to construct the
params, MotionCorrect and cnmf objects and call the relevant functions. You
can also run a large part of the pipeline with a single method (cnmf.fit_file)
See inside for details.

Demo is also available as a jupyter notebook (see demo_pipeline.ipynb)
Dataset couresy of Sue Ann Koay and David Tank (Princeton University)

This demo pertains to two photon data. For a complete analysis pipeline for
one photon microendoscopic data see demo_pipeline_cnmfE.py

copyright GNU General Public License v2.0
authors: @agiovann and @epnev
"""

import cv2
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
#import h5py
#import deepdish

try:
    cv2.setNumThreads(0)
except:
    pass

#try:
#    if __IPYTHON__:
#        # this is used for debugging purposes only. allows to reload classes
#        # when changed
#        get_ipython().magic('load_ext autoreload')
#        get_ipython().magic('autoreload 2')
#except NameError:
#    pass

import pandas as pd

import caiman as cm
from caiman.source_extraction import cnmf as cnmf
from caiman.motion_correction import MotionCorrect

#from caiman.source_extraction.cnmf import cnmf as cnmf
#from caiman.source_extraction.cnmf import params as params


#from caiman.utils.utils import download_demo

# %%
# Set up the logger (optional); change this if you like.
# You can log to a file using the filename parameter, or make the output more
# or less verbose by setting level to logging.DEBUG, logging.INFO,
# logging.WARNING, or logging.ERROR

logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"\
                    "[%(process)d] %(message)s",
                    level=logging.INFO)


# %%
def main():
    pass  # For compatibility between running under Spyder and the CLI

    #%% No moco here
    """
    General parameters
    """
    
    plot_stuff = 0;
    save_results = True;
    
    
    #%% Select file(s) to be processed (download if not present)
    """
    Load file
    """
    
    #f_dir = r'F:\\AC_data\\caiman_data_missmatch\\movies\\'
    f_dir = r'F:\\AC_data\\caiman_data_dream3\\movies\\'
    #f_dir = 'D:\\data\\caiman_data_dream\\movies\\'
    
    
    #save_dir = 'F:\\AC_data\\caiman_data_missmatch\\'
    save_dir = 'F:\\AC_data\\caiman_data_dream3\\'
    
    
    f_ext = 'h5'
    

    
    # f_name = ['AC_ammn5_11_24_21_mpl5_pl1',
    #           'AC_ammn5_11_24_21_mpl5_pl2',
    #           'AC_ammn5_11_24_21_mpl5_pl3',
    #           'AC_ammn5_11_24_21_mpl5_pl4',
    #           'AC_ammn5_11_24_21_mpl5_pl5']
    #           # 'AC_ammn1_1_31_22_mpl5_pl1',
    #           # 'AC_ammn1_1_31_22_mpl5_pl2',
    #           # 'AC_ammn1_1_31_22_mpl5_pl3',
    #           # 'AC_ammn1_1_31_22_mpl5_pl4',
    #           # 'AC_ammn1_1_31_22_mpl5_pl5'];
    
    
    # process all files in dir
    f_list = os.listdir(f_dir)
    f_name = [];
    for fil1 in f_list:
        if fil1.endswith('.h5'):
            fname2 = os.path.splitext(fil1)[0]
            if not os.path.exists(save_dir + fname2 + '_results_cnmf.hdf5'):
                f_name.append(fname2);

    
    #%%
    for n_fl in range(len(f_name)):
        
        #%%
        #fnames = [f_dir + f_name[n_fl] + '.' + f_ext]
        
        # n_fl = 0
        # f_name = [];
        # f_name.append('test')
        
        # n_processes = 4
        
        # fnames = [f_dir + 'M166_im1_AC_tone_lick_reward1_6_20_22_mpl5_pl1' + '.' + f_ext,
        #           f_dir + 'M166_im2_AC_ammn2_6_20_22_mpl5_pl1' + '.' + f_ext,
        #           f_dir + 'M166_im3_AC_rest3_6_20_22_mpl5_pl1' + '.' + f_ext,
        #           f_dir + 'M166_im4_AC_ammn_stim4_6_20_22_mpl5_pl1' + '.' + f_ext,
        #           f_dir + 'M166_im5_AC_rest5_6_20_22_mpl5_pl1' + '.' + f_ext,
        #           f_dir + 'M166_im6_AC_ammn6_6_20_22_mpl5_pl1' + '.' + f_ext]
        
        print('Running ' + fnames[0])
    
        #fnames = ['C:/Users/rylab_dataPC/Desktop/Yuriy/caiman_data/rest1_5_9_19_2_cut_ca.hdf5']
    
        #%% load mov
        # need to create memmap version of movie
        print('Saving memmap...')
        fname_mmap = cm.mmapping.save_memmap(fnames, base_name=save_dir+'\\movies\\memmap_'+f_name[n_fl], order='C', ) # 
        
        
        Yr, dims, T = cm.load_memmap(fname_mmap)   # mc.mmap_file[0]
        images = np.reshape(Yr.T, [T] + list(dims), order='F')
        del Yr
        
        
        #plt.figure();
        #plt.imshow(np.mean(images, axis=0))
        #%% restart cluster to clean up memory
        if 'dview' in locals():
            cm.stop_server(dview=dview)
        c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=n_processes, # =None
                                         single_thread=False)
    
        # %%  parameters for source extraction and deconvolution
        
        fr = 10             # imaging rate in frames per second
        decay_time = 2; # 1;#0.4    # length of a typical transient in seconds
        
        p = 2                    # order of the autoregressive system
        gnb = 2                  # number of global background components
        merge_thr = 0.85         # merging threshold, max correlation allowed
        rf = 50                  # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
        stride_cnmf = 12          # amount of overlap between the patches in pixels
        K = 30                    # number of components per patch
        gSig = [4, 4]            # expected half size of neurons in pixels
        # initialization method (if analyzing dendritic data using 'sparse_nmf')
        method_init = 'greedy_roi'
        fudge_factor = .99 # *****important for temporal inference : float (close but smaller than 1) (0< fudge_factor <= 1) default: .96 bias correction factor for discrete time constants shrinkage factor to reduce bias
        ssub = 1                     # spatial subsampling during initialization
        tsub = 1                     # temporal subsampling during intialization
        
        # parameters for component evaluation
        opts_dict = {
                     #'fnames': fnames,            
                     'fr': fr,
                     'decay_time': decay_time,
                     'nb': gnb,
                     'rf': rf,
                     'stride': stride_cnmf,
                     'K': K,
                     'gSig': gSig,
                     'fudge_factor': fudge_factor,
                     'method_init': method_init,
                     'rolling_sum': True,
                     'merge_thr': merge_thr,
                     'n_processes': n_processes,
                     'ssub': ssub,
                     'tsub': tsub}
        
        opts = cnmf.params.CNMFParams(params_dict=opts_dict)
        
        #opts.change_params(params_dict=opts_dict)
        # %% RUN CNMF ON PATCHES
        # First extract spatial and temporal components on patches and combine them
        # for this step deconvolution is turned off (p=0)
        print('Fitting cnmf...')
        
        opts.change_params({'p': 0})
        cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
        cnm = cnm.fit(images)
           
    
        # %% plot contours of found components
        Cn = cm.local_correlations(images, swap_dim=False)
        Cn[np.isnan(Cn)] = 0
        
        if plot_stuff:
            cnm.estimates.plot_contours(img=Cn)
            plt.title('Contour plots of found components')
            
        if plot_stuff:
            plt.figure()
            plt.imshow(Cn)
            plt.title('Local correlations')
    
    
        # %% RE-RUN seeded CNMF on accepted patches to refine and perform deconvolution
        print('Refitting cnmf...')
        
        cnm.params.change_params({'p': p})
        cnm2 = cnm.refit(images, dview=dview)
        
        cnm2.estimates.dims = cnm2.dims
    
        # %% COMPONENT EVALUATION
        # the components are evaluated in three ways:
        #   a) the shape of each component must be correlated with the data
        #   b) a minimum peak SNR is required over the length of a transient
        #   c) each shape passes a CNN based classifier
        min_SNR = 2  # signal to noise ratio for accepting a component
        rval_thr = 0.90 # space correlation threshold for accepting a component
        cnn_thr = 0.99  # threshold for CNN based classifier
        cnn_lowest = 0.1 # neurons with cnn probability lower than this value are rejected
        
        cnm2.params.set('quality', {'decay_time': decay_time,
                                   'min_SNR': min_SNR,
                                   'rval_thr': rval_thr,
                                   'use_cnn': True,
                                   'min_cnn_thr': cnn_thr,
                                   'cnn_lowest': cnn_lowest})
        cnm2.estimates.evaluate_components(images, cnm2.params, dview=dview)
    
        # %% PLOT COMPONENTS
        cnm2.estimates.plot_contours(img=Cn, idx=cnm2.estimates.idx_components)
        plt.suptitle('Component selection: %s min_SNR=%.2f; rval_thr=%.2f;; cnn prob range=[%.2f %.2f]' % (f_name[n_fl], min_SNR, rval_thr, cnn_lowest, cnn_thr));
        
        # %% VIEW TRACES (accepted and rejected)
        
        if plot_stuff:
            cnm2.estimates.view_components(images, img=Cn, idx=cnm2.estimates.idx_components)
            plt.suptitle('Accepted')
            cnm2.estimates.view_components(images, img=Cn, idx=cnm2.estimates.idx_components_bad)
            plt.suptitle('Rejected')
            
    
        #%% update object with selected components
        #cnm2.estimates.select_components(use_object=True)
        #%% Extract DF/F values
        cnm2.estimates.detrend_df_f(quantileMin=8, frames_window=250)
        
        #%% Show final traces
        if plot_stuff:
            cnm2.estimates.view_components(img=Cn)
            plt.suptitle("Final results")
        
        
        #%% STOP CLUSTER and clean up log files
        cm.stop_server(dview=dview)
        log_files = glob.glob('*_LOG_*')
        for log_file in log_files:
            os.remove(log_file)
        
        
        print('Saving...')
        if save_results:
            cnm2.save(save_dir + f_name[n_fl] + '_results_cnmf.hdf5')
    
    print('Done')
#%%
# This is to mask the differences between running this demo in Spyder
# versus from the CLI
if __name__ == "__main__":
    main()
