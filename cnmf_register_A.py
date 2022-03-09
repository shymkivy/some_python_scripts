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

from scipy.io import savemat
import pandas as pd
import re


#%%

data_dir = 'C:\\Users\\ys2605\\Desktop\\stuff\\AC_data\\caiman_data_dream\\'


df = pd.read_excel (r'C:\Users\ys2605\Desktop\stuff\AC_2p_analysis\AC_data_list_all.xlsx')


#%% load info

# collect dset names from xlsx file
idx1 = df['experiment'] == 'dream';
df2 = df.loc[idx1]
dset_list = [];
for n_dset in range(len(df2)):
    temp_dset = df2.iloc[n_dset]
    dset_list.append('%s_im%d_%s_%s' % (temp_dset['mouse_id'], temp_dset['im_num'], temp_dset['dset_name'], temp_dset['mouse_tag']))
    

#  collect files in provided folder
fname_list_dir = os.listdir(data_dir)
fname_list_all = [];
for fil1 in fname_list_dir:
    if fil1.endswith('.hdf5'):
        fname2 = os.path.splitext(fil1)[0]
        #if not os.path.exists(save_dir + fname2 + '_results_cnmf.hdf5'):
        fname_list_all.append(fname2);
       

# fname_list = ['A1_cont_1_12_4_21b_mpl5_pl5_results.hdf5',
#               'A1_cont_2_12_4_21b_mpl5_pl5_results.hdf5',
#               'A1_cont_4_12_4_21b_mpl5_pl5_results.hdf5',
#               'A1_cont_05_12_4_21b_mpl5_pl5_results.hdf5'];

# save_fname = 'A1_cont_12_4_21b_mpl5_pl5_reg.mat'


#%%

for n_dset in range(len(dset_list)):
    
    dset_name = dset_list[n_dset]
    
    fname_list = []
    
    for n_file in range(len(fname_list_all)):
        
        m = re.search(dset_name, fname_list_all[n_file])
        if m:
            fname_list.append(f_name[n_file]+'.hdf5')

            
        
    A_list = [];

    for fname in fname_list:
        cnm = cnmf.cnmf.load_CNMF(data_dir+fname);
        A_list.append(cnm.estimates.A)
    
    
    dims = cnm.estimates.dims;


    #out = register_ROIs(A1, A2, dims=dims);

    out = register_multisession(A_list, dims=dims)
    
    class_list_save = {"fname_list": fname_list,
                   "A_list": A_list,
                   "reg_out": list(out)}
             
    savemat(data_dir+ dset_name + '_registration.mat', class_list_save)
    
#%%



#save_fname = 'rnn_out_8_25_21_1_complex_g_tau10_5cycles.mat'



