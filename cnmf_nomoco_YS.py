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
    play_movie = 0;
    plot_extras = 0;
    plot_extras_cell = 0;
    compute_mc_metrics = 0;
    
    #%% Select file(s) to be processed (download if not present)
    """
    Load file
    """
    
    #fnames = ['Sue_2x_3000_40_-46.tif']  # filename to be processed
    #if fnames[0] in ['Sue_2x_3000_40_-46.tif', 'demoMovie.tif']:
    #    fnames = [download_demo(fnames[0])]
    
    #fnames = ['/home/yuriy/Desktop/Data/rest1_5_9_19_cut.tif']
    
    
    #f_dir = 'C:\\Users\\rylab_dataPC\\Desktop\\Yuriy\\caiman_data\\short\\'
    #f_dir = 'G:\\data\\Auditory\\caiman_out\\movies\\'
    #f_dir = 'C:\\Users\\ys2605\\Desktop\\stuff\\AC_data\\cmnf_data\\'
    #f_dir = 'G:\\analysis\\190828-calcium_voltage\\soma_dendrites\\pCAG_jREGECO1a_ASAP3_anesth_001\\'
    f_dir = 'C:\\Users\\ys2605\\Desktop\\stuff\\AC_data\\caiman_data\\movies\\'
    
    f_name = 'A1_cont_2_12_4_21b_mpl5_pl2';
    f_ext = 'h5'
    fnames = [f_dir + f_name + '.' + f_ext]
    
    
    #fnames = ['C:/Users/rylab_dataPC/Desktop/Yuriy/caiman_data/rest1_5_9_19_2_cut_ca.hdf5']
    
    #%% load mov
    # need to create memmap version of movie
    fname_mmap = cm.mmapping.save_memmap(fnames, base_name='memmap', order='C') # 
    
    
    Yr, dims, T = cm.load_memmap(fname_mmap)   # mc.mmap_file[0]
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    del Yr
    

    #plt.figure();
    #plt.imshow(np.mean(images, axis=0))
    # %% restart cluster to clean up memory
    if 'dview' in locals():
        cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None,
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
    gSig = [3, 3]            # expected half size of neurons in pixels
    # initialization method (if analyzing dendritic data using 'sparse_nmf')
    method_init = 'greedy_roi'
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
                 
                 'method_init': method_init,
                 'rolling_sum': True,
                 'merge_thr': merge_thr,
                 'n_processes': n_processes,
                 'only_init': True,
                 'ssub': ssub,
                 'tsub': tsub}
    
    opts = cnmf.params.CNMFParams(params_dict=opts_dict)
    
    #opts.change_params(params_dict=opts_dict)
    # %% RUN CNMF ON PATCHES
    # First extract spatial and temporal components on patches and combine them
    # for this step deconvolution is turned off (p=0)
    
    opts.change_params({'p': 0})
    cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
    cnm = cnm.fit(images)
   

    # %% plot contours of found components
    Cn = cm.local_correlations(images, swap_dim=False)
    Cn[np.isnan(Cn)] = 0
    cnm.estimates.plot_contours(img=Cn)
    plt.title('Contour plots of found components')
    
    if plot_extras:
        plt.figure()
        plt.imshow(Cn)
        plt.title('Local correlations')
    
    
    # %% RE-RUN seeded CNMF on accepted patches to refine and perform deconvolution
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
    
    if plot_extras:
        cnm2.estimates.view_components(images, img=Cn, idx=cnm2.estimates.idx_components)
        plt.suptitle('Accepted')
        cnm2.estimates.view_components(images, img=Cn, idx=cnm2.estimates.idx_components_bad)
        plt.suptitle('Rejected')
        
        
        
    #plt.figure();
    #plt.plot(cnm2.estimates.YrA[0,:]+cnm2.estimates.C[0,:])
    #
    #
    #
    #
    #plt.figure();
    #plt.plot(cnm2.estimates.R[0,:]-cnm2.estimates.YrA[0,:]);
    #plt.plot();
    #plt.show();
    #
    #
    #plt.figure();
    #plt.plot(cnm2.estimates.detrend_df_f[1,:])
    
    # these store the good and bad components, and next step sorts them
    # cnm2.estimates.idx_components
    # cnm2.estimates.idx_components_bad
    
    
    #%% update object with selected components
    #cnm2.estimates.select_components(use_object=True)
    #%% Extract DF/F values
    cnm2.estimates.detrend_df_f(quantileMin=8, frames_window=250)
    
    #%% Show final traces
    
    cnm2.estimates.view_components(img=Cn)
    plt.suptitle("Final results")
    
    #%% Save the mc data as in cmn struct as well
    
    ##
    #mc_out = dict(
    #            pw_rigid            = mc.pw_rigid,
    #            fname               = mc.fname,
    #            mmap_file           = mc.mmap_file,
    #            total_template_els  = mc.total_template_els,
    #            total_template_rig  = mc.total_template_rig,
    #            border_nan          = mc.border_nan,
    #            border_to_0         = mc.border_to_0,
    #            x_shifts_els        = mc.x_shifts_els,
    #            y_shifts_els        = mc.y_shifts_els,
    #            Cn                  = Cn
    #            )
    #    
    #
    #deepdish.io.save(fnames[0] + '_mc_data.hdf5', mc_out)
    
    
    #%% reconstruct denoised movie (press q to exit)
    if play_movie:
        cnm2.estimates.play_movie(images, q_max=99.9, gain_res=2, magnification=2,
                                  bpx=border_to_0, include_bck=False)  # background not shown
    
    #%% STOP CLUSTER and clean up log files
    cm.stop_server(dview=dview)
    log_files = glob.glob('*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)
    
    
    save_results = True
    if save_results:
        cnm2.save(fnames[0][:-3] + '_results_cnmf.hdf5')
    

#%%
# This is to mask the differences between running this demo in Spyder
# versus from the CLI
if __name__ == "__main__":
    main()
