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

#%% load info

df = pd.read_excel (r'C:\Users\ys2605\Desktop\stuff\AC_2p_analysis\AC_data_list_all.xlsx')


f_dir = 'D:\\data\\caiman_data_dream\\movies\\'
    

save_dir = 'C:\\Users\\ys2605\\Desktop\\stuff\\AC_data\\caiman_data_dream\\'

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


# make list of files
f_list = os.listdir(f_dir)
f_name = [];
for fil1 in f_list:
    if fil1.endswith('.hdf5'):
        fname2 = os.path.splitext(fil1)[0]
        if not os.path.exists(save_dir + fname2 + '_results_cnmf.hdf5'):
            f_name.append(fname2);
        


f_dir = 'C:\\Users\\ys2605\\Desktop\\stuff\\AC_data\\\caiman_data\\'

fname_list = ['A1_cont_1_12_4_21b_mpl5_pl5_results.hdf5',
              'A1_cont_2_12_4_21b_mpl5_pl5_results.hdf5',
              'A1_cont_4_12_4_21b_mpl5_pl5_results.hdf5',
              'A1_cont_05_12_4_21b_mpl5_pl5_results.hdf5'];

save_fname = 'A1_cont_12_4_21b_mpl5_pl5_reg.mat'


#%% load 

A_list = [];

for fname in fname_list:
    cnm = cnmf.cnmf.load_CNMF(f_dir+fname);
    A_list.append(cnm.estimates.A)


dims = cnm.estimates.dims;

#%%
#out = register_ROIs(A1, A2, dims=dims);

#%%
out = register_multisession(A_list, dims=dims)

#%%

class_list_save = {"fname_list": fname_list,
                   "A_list": A_list,
                   "reg_out": list(out)}
             

#save_fname = 'rnn_out_8_25_21_1_complex_g_tau10_5cycles.mat'


savemat(f_dir+ save_fname, class_list_save)
