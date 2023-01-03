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
from matplotlib import gridspec
import numpy as np
import scipy as sc
import os
import sys
#import h5py
#import deepdish

try:
    cv2.setNumThreads(0)
except:
    pass

#cwd = os.getcwd() 
sys.path.append('C:\\Users\\ys2605\\Desktop\\stuff\\python_scripts')
from f_caiman_extra_YS import f_merge_components, f_remove_components


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
#from caiman.motion_correction import MotionCorrect

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
                    level=logging.WARNING)


# %%
def main():
    pass  # For compatibility between running under Spyder and the CLI

    #%% No moco here
    """
    General parameters
    """
    
    plot_stuff = 0;
    save_results = True;
    
    n_proc = 6; # or None to use all
    
    combine_same_fov = False;
    
    do_merge = True;
    
    #%% Select file(s) to be processed (download if not present)
    """
    Load file
    """
    
    #f_dir = 'F:\\AC_data\\caiman_data_missmatch\\movies\\'
    f_dir = 'E:\\data\\AC\\caiman_data_echo\\movies\\'
    #f_dir = 'H:\\data\\caiman_data_dream\\movies\\'
    
    
    #save_dir = 'F:\\AC_data\\caiman_data_missmatch\\'
    save_dir = 'F:\\AC_data\\caiman_data_echo\\'
    #save_dir = 'F:\\AC_data\\caiman_data_dream\\'
    
    f_ext = 'h5'

    df = pd.read_excel (r'C:\Users\ys2605\Desktop\stuff\AC_2p_analysis\AC_data_list_all.xlsx')

    df = df.loc[~pd.isnull(df.mouse_id)]
    
    df = df.loc[~pd.isnull(df.do_proc)]
    
    df = df.loc[df.do_proc == 1]
    
    experiment_tag = 'echo'
    mouse_id = ''
    mouse_tag = ''
    
    
    if len(experiment_tag):
        df = df.loc[df.experiment == experiment_tag]
        
    if len(mouse_tag):
        df = df.loc[df.mouse_tag == mouse_tag]
        
    if len(mouse_id):
        df = df.loc[df.mouse_id == mouse_id]
    
    f_list = os.listdir(f_dir)
    
#%% from excel file generate list of inputs and save names

    dsets_to_run = [];
    dsets_names = [];
    
    mouse_tag_all = df.mouse_tag.unique()
    
    df_all = [];
    
    missing_files = [];
    missing_files_num = [];
    
    for n_ms in range(len(mouse_tag_all)):
        ms_tag = mouse_tag_all[n_ms]
        df2 = df.loc[df.mouse_tag == ms_tag]
        
        fov_all = df2.FOV_num.unique();
        for n_fov in range(len(fov_all)):
            fov1 = fov_all[n_fov]
            df3 = df2.loc[df2.FOV_num == fov1]
            
            n_idx1 = df3.index[0]
            mpl1 = round(df3.mpl[n_idx1])
            
            mpl_tag = [];
            if mpl1 > 1:
                for n_pl in range(mpl1):
                    mpl_tag.append('_mpl%d_pl%d' % (mpl1, n_pl+1))
            else:
                mpl_tag.append('');
            
            for n_pl in range(len(mpl_tag)):
                
                file_in_fov = [];
                missing_files2 = 0;
                
                for n_fil in range(len(df3.dset_name)):
                    
                    n_idx2 = df3.index[n_fil]
   
                    fname1 = '%s_im%d_%s_%s%s' % (df3.mouse_id[n_idx2], df3.im_num[n_idx2], df3.dset_name[n_idx2], df3.mouse_tag[n_idx2], mpl_tag[n_pl])
                    fname2 = fname1 + '.' + f_ext
                    fname3 = f_dir + fname2;
                    
                    if combine_same_fov:
                        file_in_fov.append(fname3)
                        if fname2 not in f_list:
                            print('File %s is missing from data ' % fname2)
                            missing_files.append(fname2);
                            missing_files2 += 1;
                    else:
                        dsets_to_run.append([fname3])
                        dsets_names.append(fname1)
                        df_all.append(df3.loc[df3.index == n_idx2]);
                        if fname2 not in f_list:
                            print('File %s is missing from data ' % fname2)
                            missing_files_num.append(1)
                        else:
                            missing_files_num.append(0)
                            
                if combine_same_fov:
                    missing_files_num.append(missing_files2)
                
                if combine_same_fov:
                    dset_name1 = '%s_fov%d_%s_im%d_to_im%d%s' % (df3.mouse_id[n_idx1], df3.FOV_num[n_idx1], df3.mouse_tag[n_idx2], df3.im_num[n_idx1], df3.im_num[n_idx2],mpl_tag[n_pl])
                    dsets_to_run.append(file_in_fov)
                    dsets_names.append(dset_name1)
                    df_all.append(df3)
                    
    #%%

    for n_dset in range(len(dsets_names)): #
        
        save_name = dsets_names[n_dset]
        print(n_dset)
        if not os.path.exists(save_dir + save_name + '_results_cnmf.hdf5'):
            if not missing_files_num[n_dset]:
                
                print('Running %d of %d; %s' % (n_dset, len(dsets_names), save_name))
                
                fnames = dsets_to_run[n_dset];
                
                df_curr = df_all[n_dset]
                n_idx = df_curr.index[0]
                
                #fnames = ['C:/Users/rylab_dataPC/Desktop/Yuriy/caiman_data/rest1_5_9_19_2_cut_ca.hdf5']
            
                #%% load mov
                # need to create memmap version of movie
                print('Saving memmap...')
                fname_mmap = cm.mmapping.save_memmap(fnames, base_name=save_dir+'\\movies\\memmap_'+save_name, order='C', ) # 
                
                
                Yr, dims, T = cm.load_memmap(fname_mmap)   # mc.mmap_file[0]
                images = np.reshape(Yr.T, [T] + list(dims), order='F')
                del Yr
                
                #plt.figure();
                #plt.imshow(np.mean(images, axis=0))
                #%% restart cluster to clean up memory
                if 'dview' in locals():
                    cm.stop_server(dview=dview)
                c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=n_proc,
                                                 single_thread=False)
            
                # %%  parameters for source extraction and deconvolution
                
                if df_curr.mpl[n_idx] == 1:
                    fr = 30             # imaging rate in frames per second
                    K = 50                    # number of components per patch
                elif df_curr.mpl[n_idx] == 5:
                    fr = 10             # imaging rate in frames per second
                    K = 30                    # number of components per patch
                else:
                    raise NameError('number mpl not defined for %s' % save_name)
                
                if df_curr.obj[n_idx].lower() == '20x'.lower():
                    if df_curr.im_zoom[n_idx] <= 1:
                        gSig = [3, 3]            # expected half size of neurons in pixels
                    elif df_curr.im_zoom[n_idx] > 1 and df_curr.im_zoom[n_idx] <= 1.41:
                        gSig = [4, 4]            # expected half size of neurons in pixels
                    else:
                        gSig = [5, 5]            # expected half size of neurons in pixels
                elif df_curr.obj[n_idx].lower() == '25x'.lower():
                    if df_curr.im_zoom[n_idx] <= 1:
                        gSig = [4, 4]            # expected half size of neurons in pixels
                    elif df_curr.im_zoom[n_idx] > 1 and df_curr.im_zoom[n_idx] <= 1.41:
                        gSig = [5, 5]            # expected half size of neurons in pixels
                    else:
                        raise NameError('zoom params not optimized for %s' % save_name)
                else:
                    raise NameError('Objective name not found for %s' % save_name)
                
    
                decay_time = 2; # 1;#0.4    # length of a typical transient in seconds
                
                p = 2                    # order of the autoregressive system
                gnb = 2                  # number of global background components
                nb_patch = 0
                merge_thr = 0.85         # merging threshold, max correlation allowed
                rf = 40                  # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
                stride_cnmf = 20          # amount of overlap between the patches in pixels
                
                
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
                             'low_rank_background': True,
                             'nb_patch': 2,
                             'rf': rf,
                             'stride': stride_cnmf,
                             'K': K,
                             'gSig': gSig,
                             'fudge_factor': fudge_factor,
                             'method_init': method_init,
                             'rolling_length':     round(1000/fr),    # temporal mean smooth frames
                             #'rolling_sum': True,
                             'merge_thr': merge_thr,
                             'nIter': 10,
                             'n_processes': n_processes,
                             'ssub': ssub,
                             'tsub': tsub,
                             #'remove_very_bad_comps':   True, this gets overwritten
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
                
                print('After first fit: %d comp total' % (cnm.estimates.C.shape[0]))
                
                #%%
                if do_merge:
                    f_merge_components(images, cnm)
                
                # %% COMPONENT EVALUATION
                # the components are evaluated in three ways:
                #   a) the shape of each component must be correlated with the data
                #   b) a minimum peak SNR is required over the length of a transient
                #   c) each shape passes a CNN based classifier
                min_SNR = 1  # signal to noise ratio for accepting a component
                rval_thr = 0.5 # space correlation threshold for accepting a component
                cnn_thr = 0  #  .1 threshold for CNN based classifier
                cnn_lowest = 0 #  .9 neurons with cnn probability lower than this value are rejected
                
                cnm.params.set('quality', {'decay_time': decay_time,
                                           'min_SNR': min_SNR,
                                           'rval_thr': rval_thr,
                                           'use_cnn': True,
                                           'min_cnn_thr': cnn_thr,
                                           'cnn_lowest': cnn_lowest})
                
                cnm.estimates.evaluate_components(images, cnm.params, dview=dview)
                
                # remove very bad comps
                f_remove_components(cnm, cnm.estimates.idx_components_bad)
                #cnm.estimates.select_components(use_object=True, save_discarded_components=False)
                print('First bad comp removal: %d comp total' % (cnm.estimates.C.shape[0]))  
                
                # %% RE-RUN seeded CNMF on accepted patches to refine and perform deconvolution
                print('Refitting cnmf...')
                
                cnm.params.change_params({'p': p})
                cnm2 = cnm.refit(images, dview=dview)
                cnm2.estimates.dims = cnm2.dims
                
                print('First refit: %d comp total' % (cnm2.estimates.C.shape[0]))     
                
                #%%
              
                min_SNR = 2  # signal to noise ratio for accepting a component
                rval_thr = 0.6 # space correlation threshold for accepting a component
                cnn_thr = .1  #  .1 threshold for CNN based classifier
                cnn_lowest = .9 #  .9 neurons with cnn probability lower than this value are rejected
                
                cnm2.params.set('quality', {'decay_time': decay_time,
                                           'min_SNR': min_SNR,
                                           'rval_thr': rval_thr,
                                           'use_cnn': True,
                                           'min_cnn_thr': cnn_thr,
                                           'cnn_lowest': cnn_lowest})
                
                cnm2.estimates.evaluate_components(images, cnm2.params, dview=dview)
                
                print('After final quality est: %d comp total' % (cnm2.estimates.C.shape[0]))
                
                #%% Extract DF/F values
                cnm2.estimates.detrend_df_f(quantileMin=8, frames_window=250)            
                
                # %% PLOT COMPONENTS

                if plot_stuff:
                    cnm_plot = cnm2
                    
                    
                    Cn = cm.local_correlations(images, swap_dim=False)
                    Cn[np.isnan(Cn)] = 0
                    
                    plt.figure()
                    plt.imshow(Cn)
                    
                    cnm_plot.estimates.plot_contours(img=Cn)
                    plt.title('Contour plots of found components')
                    
                    cnm_plot.estimates.plot_contours(img=Cn, idx=cnm_plot.estimates.idx_components)
                    plt.suptitle('Component selection: %s min_SNR=%.2f; rval_thr=%.2f;; cnn prob range=[%.2f %.2f]' % (save_name, min_SNR, rval_thr, cnn_lowest, cnn_thr));
                    
                    # VIEW TRACES (accepted and rejected)
                
                    cnm_plot.estimates.view_components(images, img=Cn, idx=cnm_plot.estimates.idx_components)
                    plt.suptitle('Accepted')
                    cnm_plot.estimates.view_components(images, img=Cn, idx=cnm_plot.estimates.idx_components_bad)
                    plt.suptitle('Rejected')
                    
                    # plt.close('all')
                    
                    # plt.figure()
                    # plt.imshow(cnm.estimates.C)
                    
                    # plt.figure()
                    # plt.imshow(cnm.estimates.A.sum(axis=1).reshape(cnm.dims))
                    
                    
                    # plt.figure()
                    # plt.plot(cnm.estimates.f.T)
                    
                    # plt.figure()
                    # plt.imshow(cnm.estimates.b[:,0].reshape(cnm.dims))
                    # plt.figure()
                    # plt.imshow(cnm.estimates.b[:,3].reshape(cnm.dims))
                
                    
                #%% update object with selected components
                #cnm3.estimates.select_components(use_object=True)
                
                
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
                    #cnm.save(save_dir + save_name + '_results_cnmf_cnm1.hdf5')
                    cnm2.save(save_dir + save_name + '_results_cnmf.hdf5')
            else:
                print('missing %d files in this dset' % missing_files_num[n_dset])
        else:
            print(save_name+ ' cnmf output already exists')
        
    print('Done')
#%%
# This is to mask the differences between running this demo in Spyder
# versus from the CLI
if __name__ == "__main__":
    main()
