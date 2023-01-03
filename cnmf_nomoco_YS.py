
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
import sys
#import h5py
#import deepdish

try:
    cv2.setNumThreads(0)
except:
    pass

cwd = os.getcwd() 
sys.path.append('C:\\Users\\ys2605\\Desktop\\stuff\\python_scripts')
from f_caiman_extra_YS import *

#try:
#    if __IPYTHON__:
#        # this is used for debugging purposes only. allows to reload classes
#        # when changed
#        get_ipython().magic('load_ext autoreload')
#        get_ipython().magic('autoreload 2')
#except NameError:
#    pass


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
                    level=logging.ERROR)


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
    
    f_dir = 'G:\\data\\Auditory\\caiman_data_missmatch\\movies\\'
    
    save_dir = 'F:\\AC_data\\caiman_data_missmatch\\'
    
    f_name = 'M1_im1_A1_ammn1_10_2_18';
    f_ext = 'h5'
    fnames = [f_dir + f_name + '.' + f_ext]
    
    save_tag = '' # _cvxpy
    
    
    n_processes_set = 4;
    
    #fnames = ['C:/Users/rylab_dataPC/Desktop/Yuriy/caiman_data/rest1_5_9_19_2_cut_ca.hdf5']
    
    print('Running ' + f_name)
    
    #%% load mov
    print('Saving memmap...')
    # need to create memmap version of movie
    fname_mmap = cm.mmapping.save_memmap(fnames, base_name=save_dir+'\\movies\\memmap_' + f_name, order='C') # 
    
    
    Yr, dims, T = cm.load_memmap(fname_mmap)   # mc.mmap_file[0]
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    del Yr
    
    
    #plt.figure();
    #plt.imshow(np.mean(images, axis=0))
    #%% restart cluster to clean up memory
    if 'dview' in locals():
        cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=n_processes_set,
                                      single_thread=False)
    
    # %%  parameters for source extraction and deconvolution
    
    fr = 30             # imaging rate in frames per second
    decay_time = 2; # 1;#0.4    # length of a typical transient in seconds
    
    p = 2                    # order of the autoregressive system
    gnb = 2                  # number of global background components
    merge_thr = 0.85         # merging threshold, max correlation allowed
    rf = 50                  # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
    stride_cnmf = 12          # amount of overlap between the patches in pixels
    K = 30                    # number of components per patch
    gSig = [4, 4]            # ******important, must be large enought so roi not split. expected half size of neurons in pixels
    # initialization method (if analyzing dendritic data using 'sparse_nmf')
    method_init = 'greedy_roi' # greedy_roi, corr_pnr, sparse_NMF, local_NMF
    fudge_factor = .99 # *****important for temporal inference : float (close but smaller than 1) (0< fudge_factor <= 1) default: .96 bias correction factor for discrete time constants shrinkage factor to reduce bias
    # nrgthr = .999           # float, default: 0.9999 Energy threshold for spatial comp .999 maybe looks little better but not much diff
    # rolling_sum  # for greedy roi, False, True
    # maxIter = 10; # % default 5; number HALS iter during init
    # Iter = 10 #  int, default: 5 number of rank-1 refinement iterations during greedy_roi initialization
    # method_ls = 'nnls_L0' #default: ‘lasso_lars’ ‘nnls_L0’. Nonnegative least square with L0 penalty ‘lasso_lars’ lasso lars function from scikit learn broken
    # optimize_g = True     # flag for optimizing time constants' default false - gives error
    # SC_kernel = 'cos'
    # block_size = 10000 # 5000 default
    # nb_patch = 2;
    # lambda_gnmf = 1; #  float, default: 1. regularization weight for graph NMF - seems to do nothing 
    # method_exp = 'ellipse' # ‘dilate’|’ellipse’, default: ‘dilate’ method for expanding footprint of spatial components
    
    ssub = 1                     # spatial subsampling during initialization
    tsub = 1                     # temporal subsampling during intialization
    #method_deconvolution = 'cvxpy' # oasis
    # lambda_gnmf # regularization weight NMF, default 1
    # maxIter # numner of hals iterations during init, default 5
    
    # parameters for component evaluation
    opts_dict = {
                 #'fnames':                  fnames,                
                 'fr':                      fr,
                 'decay_time':              decay_time,
                 'nb':                      gnb,
                 #'rf':                      rf,
                 #'stride':                  stride_cnmf,
                 'K':                       K,
                 'gSig':                    gSig,
                 'fudge_factor':            fudge_factor,
                 'method_init':             method_init,
                 'rolling_length':          round(1000/fr),    # temporal mean smooth frames
                 #'nrgthr':                  nrgthr,
                 #'ring_size_factor':        2,
                 'merge_thr':               merge_thr,
                 'n_processes':             n_processes,
                 'ssub':                    ssub,
                 'tsub':                    tsub,
                 'remove_very_bad_comps':   True,
                 }
    
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
    plt.suptitle('Component selection: min_SNR=' + str(min_SNR) + '; rval_thr=' + str(rval_thr) + '; cnn prob range=[' + str(cnn_lowest) + ' ' + str(cnn_thr) + ']');
    
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
        cnm2.save(save_dir + f_name + save_tag + '_results_cnmf.hdf5')
    
    print('Done')
#%%
# This is to mask the differences between running this demo in Spyder
# versus from the CLI
if __name__ == "__main__":
    main()
