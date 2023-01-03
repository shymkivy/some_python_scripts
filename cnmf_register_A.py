# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 11:52:00 2022

@author: ys2605
"""
import os
import caiman as cm
from caiman.paths import caiman_datadir
from caiman.source_extraction import cnmf as cnmf

from caiman.base.rois import register_ROIs
from caiman.base.rois import register_multisession

from scipy.io import savemat, loadmat
import h5py
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt


#%%

#data_dir = 'F:\\AC_data\\caiman_data_dream\\'
data_dir = 'F:\\AC_data\\caiman_data_missmatch\\'


df = pd.read_excel (r'C:\Users\ys2605\Desktop\stuff\AC_2p_analysis\AC_data_list_all.xlsx')

overwrite_saved = False

#%% load info

experiment = 'missmatch';

# collect dset names from xlsx file
df2 = df[df['experiment'] == experiment]
df3 = df2[df2['do_proc'] == 1]

mouse_list = np.unique(df3['mouse_id'].values);



#%%

fname_list_dir = os.listdir(data_dir)

for n_mouse in range(len(mouse_list)):
    mouse_id = mouse_list[n_mouse]
    
    df4 = df3[df3['mouse_id'] == mouse_id]
    
    FOVs = np.unique(df4['FOV_num'])
    
    for n_fov in range(len(FOVs)):
        FOV1 = FOVs[n_fov]
        
        df5 = df4[df4['FOV_num'] == FOV1]
        
        if len(df5) > 1:
            num_planes = np.unique(df5['mpl']);
            if len(num_planes) > 1:
                raise ValueError('number of planes needs to be constant in %s, FOV%d' % (mouse_id, FOV1))
            if isinstance(num_planes, np.ndarray):
                num_planes = num_planes[0]
            
            for n_pl in range(1, int(num_planes+1)):
                
                dset_out_name = '%s_fov%d_%s' % (mouse_id, FOV1, df5.iloc[0]['mouse_tag'])
                if num_planes > 1:
                    dset_out_name = '%s_mpl%d_pl%d' % (dset_out_name, num_planes, n_pl)
                dset_out_save_name = dset_out_name + '_registration_cmnf.mat';
                
                run_plane = True
                num_dsets = len(df5)
                
                # search for all h5 and mat files in foc listm and check if everything is there
                fname_list_h = [];
                fname_list_m = [];
                dset_name_all = [];
                for n_dset in range(num_dsets):
                    #temp_dset = df5[df5.index == df5.index[n_dset]]
                    temp_dset = df5.iloc[n_dset]
                    
                    dset_name = '%s_im%d_%s_%s' % (temp_dset['mouse_id'], temp_dset['im_num'], temp_dset['dset_name'], temp_dset['mouse_tag']);
                    if num_planes > 1:
                        dset_name = '%s_mpl%d_pl%d' % (dset_name, num_planes, n_pl);    
                    
                    dset_name_all.append(dset_name);
                    
                    for n_file in range(len(fname_list_dir)):
                        
                        match1 = re.search(dset_out_save_name, fname_list_dir[n_file])
                        if match1:
                            print('File already exists %s' % (dset_out_save_name))
                            if not overwrite_saved:
                                run_plane = False
                            else:
                                print('will overwrite');
                            
                        match1 = re.search(dset_name, fname_list_dir[n_file])
                        if match1:
                            match2 = re.search('results_cnmf.hdf5', fname_list_dir[n_file])
                            if match2:
                                fname_list_h.append(fname_list_dir[n_file])
                            match2 = re.search('results_cnmf_sort.mat', fname_list_dir[n_file])
                            if match2:
                                fname_list_m.append(fname_list_dir[n_file])
                
                if len(fname_list_h) != num_dsets:
                    print('h5 file missing for %s, FOV%d' % (mouse_id, FOV1))
                    run_plane = False
                if len(fname_list_m) != num_dsets:
                    print('mat file missing for %s, FOV%d' % (mouse_id, FOV1))
                    run_plane = False
                
                if run_plane:
                    print('Registering %s, FOV%d, plane%d' % (mouse_id, FOV1, n_pl))
                    A_list = [];
                    templates_list = [];
                    for n_dset in range(len(df5)):
                    
                        cnm = cnmf.cnmf.load_CNMF(data_dir+fname_list_h[n_dset]);
                        
                        mat_file = h5py.File(data_dir+fname_list_m[n_dset], 'r');
                        comp_accepted = np.asarray(mat_file['proc']['comp_accepted'])[0].astype(bool)
                        
                        dims = cnm.dims
                        A = cnm.estimates.A[:,comp_accepted];
                        C_mean = np.mean(cnm.estimates.C, 1);
                        YrA_mean = np.mean(cnm.estimates.YrA, 1);
                        b = cnm.estimates.b;
                        f_mean = np.mean(cnm.estimates.f, 1)
                        
                        A_list.append(A)
    
                        im1 = np.reshape(np.dot(cnm.estimates.A.toarray(), C_mean + YrA_mean), (dims[0], dims[1]))
                        bkg1 = np.reshape(np.dot(b, f_mean), (dims[0], dims[1]));
                        ave_im = bkg1 + im1;
                        
                        templates_list.append(ave_im)
                        
                        #plt.figure();
                        #plt.imshow(ave_im)
            
    
                    #out = register_ROIs(A1, A2, dims=dims);
                
                    out = register_multisession(A_list, dims=dims, templates=templates_list)
                    
                    class_list_save = {"fname_list": dset_name_all,
                                       "A_list": A_list,
                                       "templates_list": templates_list,
                                       "reg_out": list(out)}
                     
                    savemat(data_dir+ dset_out_save_name, class_list_save)
                    print('Saved ' + dset_out_save_name)
                else:
                    print('Skipping ' + dset_out_save_name)
                
print('All done')
#%%
# dset_list = [];
# for n_dset in range(len(df3)):
#     temp_dset = df3.iloc[n_dset]
#     dset_list.append('%s_im%d_%s_%s' % (temp_dset['mouse_id'], temp_dset['im_num'], temp_dset['dset_name'], temp_dset['mouse_tag']))
    

# #  collect files in provided folder
# fname_list_dir = os.listdir(data_dir)
# fname_list_all = [];
# for fil1 in fname_list_dir:
#     if fil1.endswith('.hdf5'):
#         fname2 = os.path.splitext(fil1)[0]
#         #if not os.path.exists(save_dir + fname2 + '_results_cnmf.hdf5'):
#         fname_list_all.append(fname2);
       

# fname_list = ['A1_cont_1_12_4_21b_mpl5_pl5_results.hdf5',
#               'A1_cont_2_12_4_21b_mpl5_pl5_results.hdf5',
#               'A1_cont_4_12_4_21b_mpl5_pl5_results.hdf5',
#               'A1_cont_05_12_4_21b_mpl5_pl5_results.hdf5'];

# save_fname = 'A1_cont_12_4_21b_mpl5_pl5_reg.mat'


#%%

# for n_dset in range(len(dset_list)):
    
#     dset_name = dset_list[n_dset]
    
#     fname_list = []
    
#     for n_file in range(len(fname_list_all)):
        
#         m = re.search(dset_name, fname_list_all[n_file])
#         if m:
#             fname_list.append(fname_list_all[n_file]+'.hdf5')
    
#     if len(fname_list):
            
#         A_list = [];
#         templates_list = [];
    
#         for fname in fname_list:
#             cnm = cnmf.cnmf.load_CNMF(data_dir+fname);
#             A_list.append(cnm.estimates.A)
            
#             d1, d2 = cnm.dims
#             A = cnm.estimates.A;
#             C_mean = np.mean(cnm.estimates.C, 1);
#             YrA_mean = np.mean(cnm.estimates.YrA, 1);
#             b = cnm.estimates.b;
#             f_mean = np.mean(cnm.estimates.f, 1)
               
#             im1 = np.reshape(np.dot(A.toarray(), C_mean + YrA_mean), (d1, d2))
#             bkg1 = np.reshape(np.dot(b, f_mean), (d1, d2));
#             ave_im = bkg1 + im1;
            
#             templates_list.append(ave_im)
            
#             #plt.figure();
#             #plt.imshow(ave_im)
        
        
#         dims = cnm.estimates.dims;
    
    
#         #out = register_ROIs(A1, A2, dims=dims);
    
#         out = register_multisession(A_list, dims=dims, templates=templates_list)
        
#         class_list_save = {"fname_list": fname_list,
#                        "A_list": A_list,
#                        "templates_list": templates_list,
#                        "reg_out": list(out)}
                 
#         savemat(data_dir+ dset_name + '_registration.mat', class_list_save)
#         print('saved '+data_dir+ dset_name + '_registration.mat')
#%%



#save_fname = 'rnn_out_8_25_21_1_complex_g_tau10_5cycles.mat'



