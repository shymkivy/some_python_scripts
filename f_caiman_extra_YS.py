# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 17:16:27 2022

@author: ys2605
"""

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from matplotlib import gridspec


#%%

def f_remove_components(cnm, rem_cells):
    if len(rem_cells):
        C1 = cnm.estimates.C;
        C2 = np.delete(cnm.estimates.C, rem_cells, axis=0)
        cnm.estimates.C = C2;
        
        A1 = np.asarray(cnm.estimates.A.todense())
        A2 = np.delete(A1, rem_cells, axis=1)
        cnm.estimates.A = sc.sparse.csr_matrix(A2);
        
        YrA1 = cnm.estimates.YrA;
        YrA2 = np.delete(YrA1, rem_cells, axis=0)
        cnm.estimates.YrA = YrA2;
        
        cnm.estimates.idx_components = None;
        cnm.estimates.idx_components_bad = None;
        
        if hasattr(cnm.estimates, 'r_values'):
            if cnm.estimates.r_values is not None:
                r_values1 = np.asarray(cnm.estimates.r_values)
                r_values2 = np.delete(r_values1, rem_cells)
                cnm.estimates.r_values = r_values2;
            
        if hasattr(cnm.estimates, 'SNR_comp'):
            if cnm.estimates.SNR_comp is not None:
                SNR1 = np.asarray(cnm.estimates.SNR_comp)
                SNR2 = np.delete(SNR1, rem_cells)
                cnm.estimates.SNR_comp = SNR2;
        
        if hasattr(cnm.estimates, 'cnn_preds'):
            if cnm.estimates.cnn_preds is not None:
                cnn_preds1 = np.asarray(cnm.estimates.cnn_preds)
                cnn_preds2 = np.delete(cnn_preds1, rem_cells)
                cnm.estimates.cnn_preds = cnn_preds2;
            
        if hasattr(cnm.estimates, 'R'):
            if cnm.estimates.R is not None:
                cnm.estimates.R = cnm.estimates.YrA
            
        if hasattr(cnm.estimates, 'S'):
            if cnm.estimates.S is not None:
                S1 = np.asarray(cnm.estimates.S)
                S2 = np.delete(S1, rem_cells, axis=0)
                cnm.estimates.S = S2;
        
        if hasattr(cnm.estimates, 'F_dff'):
            if cnm.estimates.F_dff is not None:
                F_dff1 = np.asarray(cnm.estimates.F_dff)
                F_dff2 = np.delete(F_dff1, rem_cells, axis=0)
                cnm.estimates.F_dff = F_dff2;
                  
        if hasattr(cnm.estimates, 'g'):
            if cnm.estimates.g is not None:
                g1 = np.asarray(cnm.estimates.g)
                g2 = np.delete(g1, rem_cells)
                cnm.estimates.g = g2;
        
        if hasattr(cnm.estimates, 'bl'):
            if cnm.estimates.bl is not None:
                bl1 = np.asarray(cnm.estimates.bl)
                bl2 = np.delete(bl1, rem_cells)
                cnm.estimates.bl = bl2;
        
        if hasattr(cnm.estimates, 'c1'):
            if cnm.estimates.c1 is not None:
                c11 = np.asarray(cnm.estimates.c1)
                c12 = np.delete(c11, rem_cells)
                cnm.estimates.c1 = c12;
                
        if hasattr(cnm.estimates, 'neurons_sn'):
            if cnm.estimates.neurons_sn is not None:
                neurons_sn1 = np.asarray(cnm.estimates.neurons_sn)
                neurons_sn2 = np.delete(neurons_sn1, rem_cells)
                cnm.estimates.neurons_sn = neurons_sn2;
                
        if hasattr(cnm.estimates, 'lam'):
            if cnm.estimates.lam is not None:
                lam1 = np.asarray(cnm.estimates.lam)
                lam2 = np.delete(lam1, rem_cells)
                cnm.estimates.lam = lam2;
                
        
        print('removed %d components' % len(rem_cells))
        
        


#%%




def f_merge_components(images, cnm):
    A_corr_thresh = .3
    trace_corr_thresh = .4
    method_wa = 1; # 1 for weighted ave; 0 for full reconstruction corr
    
    est = cnm.estimates
    
    A = est.A
    C = est.C
    YrA = est.YrA
    
    dims = cnm.dims
    num_cells, T = C.shape;
    
    A_corr2 = A.T * A
    A_corr = sc.sparse.triu(A_corr2)
    A_corr.setdiag(0)
    A_corr = A_corr.todense()
    
    traces = C + YrA
    traces_corr = 1 - sc.spatial.distance.cdist(traces, traces, metric='correlation')
    np.fill_diagonal(traces_corr, 0);
    traces_corr = np.triu(traces_corr)
    
    double_corr = np.multiply((A_corr > A_corr_thresh),(traces_corr > trace_corr_thresh))
    
    idx1 = np.where(double_corr)
    
    num_match = len(idx1[0]);
    
    A_new = np.zeros((dims[0]*dims[1], num_match))
    C_new = np.zeros((num_match, T))
    YrA_new = np.zeros((num_match, T))
    rem_cells = []
    
    for n_match in range(num_match):
        n_cell1 = idx1[0][n_match]
        n_cell2 = idx1[1][n_match]
        
        A1f = A[:,n_cell1]
        A2f = A[:,n_cell2]
        A1_sum = np.sum(A1f)
        A2_sum = np.sum(A2f)
        
        trace1 = traces[n_cell1,:]
        trace2 = traces[n_cell2,:]
        trace1_sum = np.sum(trace1)
        trace2_sum = np.sum(trace2)
        
        fields1 = ['S', 'F_dff', 'SNR_comp', 'cnn_preds', 'g', 'bl', 'c1', 'neurons_sn', 'lam']
        for n_fl in range(len(fields1)):
            field1 = fields1[n_fl]
            if hasattr(est, field1):
                field_data = getattr(est, field1);
                if field_data is not None:
                    if len(field_data.shape) == 1:
                        if type(field_data[n_cell1]) is list:
                            new_stack = np.hstack((field_data, 1))
                            new_stack[-1] = []
                            setattr(est, field1, new_stack)
                        elif type(field_data[n_cell1]) == np.ndarray:
                            new_data = np.mean(np.vstack((field_data[n_cell1]*A1_sum, field_data[n_cell2]*A2_sum)), axis=0)/(A1_sum+A2_sum);
                            setattr(est, field1, np.hstack((field_data, new_data)))
                        elif field_data[n_cell1] is None:
                            setattr(est, field1, np.hstack((field_data, None)))
                        else:
                            new_data = (field_data[n_cell1]*A1_sum + field_data[n_cell2]*A2_sum)/(A1_sum + A2_sum)
                            setattr(est, field1, np.hstack((field_data, new_data)))
                    else:
                        new_data = (field_data[n_cell1,:]*A1_sum + field_data[n_cell2,:]*A2_sum)/(A1_sum + A2_sum)
                        setattr(est, field1, np.vstack((field_data, new_data)))

        if method_wa: # # weighted averages; faster
            trace_comb = (trace1*A1_sum + trace2*A2_sum)/(A1_sum + A2_sum)
            A_comb = (A1f*trace1_sum + A2f*trace2_sum)/(trace1_sum + trace2_sum)
            A_comb_norm = np.linalg.norm(A_comb.todense())
            
            A_combn = (A_comb/A_comb_norm).todense()
            trace_comb_n = trace_comb*A_comb_norm*2  # 2 because we are combining
        else:
            A_idx = ((A1f + A2f) > 0)
            
            full1 = np.dot(A1f[A_idx].T,np.reshape(trace1, (1,T)))
            full2 = np.dot(A2f[A_idx].T,np.reshape(trace2, (1,T)))
            
            full_comb = full1 + full2
            full_trace1 = np.sum(full_comb, axis=0)
            full_trace1n = full_trace1/np.linalg.norm(full_trace1)
            
            full_A = np.dot(full_comb,np.reshape(full_trace1n,(T,1)))
            full_A_norm = np.linalg.norm(full_A)
            full_An = (full_A/full_A_norm)
            
            A_combn = np.zeros((dims[0]*dims[1],1))
            A_combn[np.asarray(A_idx.todense()).flatten()] = full_An
            
            trace_comb_n = np.asarray(full_trace1n*full_A_norm).reshape((T))
        
        
        A_new[:,n_match] = A_combn.flatten()
        C_new[n_match,:] = trace_comb_n
        rem_cells.append(n_cell1)
        rem_cells.append(n_cell2)
        
        A1_2d = np.reshape(A1f.todense(),(dims[0], dims[1]))
        A2_2d = np.reshape(A2f.todense(),(dims[0], dims[1]))
        A_combn_2d = np.reshape(A_combn,(dims[0], dims[1]))
        
        idx2 = np.where(A_combn_2d)
        
        A_cat = np.concatenate((A1_2d, A2_2d, A_combn_2d))
        c_lim = [np.min(A_cat), np.max(A_cat)]
        
        y_lim = [np.min(idx2[0]), np.max(idx2[0])];
        x_lim = [np.min(idx2[1]), np.max(idx2[1])];
        
        if 1: # plot_stuff
            fig = plt.figure()
            gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1])
            ax1 = plt.subplot(gs[0])
            ax1.imshow(A1_2d, vmin=c_lim[0], vmax=c_lim[1]);ax1.set_xlim(x_lim); ax1.set_ylim(y_lim);
            ax1.set_title('cell%d' % (n_cell1))
            ax2 = plt.subplot(gs[1])
            ax2.imshow(A2_2d, vmin=c_lim[0], vmax=c_lim[1]); ax2.set_xlim(x_lim); ax2.set_ylim(y_lim)
            ax2.set_title('cell%d' % (n_cell2))
            ax3 = plt.subplot(gs[2])
            ax3.imshow(A_combn_2d, vmin=c_lim[0], vmax=c_lim[1]); ax3.set_xlim(x_lim); ax3.set_ylim(y_lim)
            ax3.set_title('A_corr=%.2f; trace_corr=%.2f' % (A_corr2[n_cell1, n_cell2], traces_corr[n_cell1, n_cell2]))
            ax4 = plt.subplot(gs[3:6])
            ax4.plot(trace1, linewidth=.5);
            ax4.plot(trace2, linewidth=.5);
            ax4.plot(trace_comb_n, linewidth=.5, color='k')
            ax4.set_title('Traces')
            ax4.autoscale(enable=True, axis='x', tight=True)
            ax4.legend(['cell1', 'cell2', 'comb'])
            fig.suptitle('merging components; method%d' % method_wa)
    
    C3 = np.vstack((C, C_new))
    
    A1 = np.asarray(A.todense())
    A3 = np.hstack((A1, A_new))
    
    YrA3 = np.vstack((YrA, YrA_new))
    
    cnm.estimates.A = sc.sparse.csr_matrix(A3)
    cnm.estimates.C = C3
    cnm.estimates.YrA = YrA3
    cnm.estimates.R = YrA3
    if est.idx_components is None:
        idx_comp = list(range(num_cells));
    for rem_cell in rem_cells:
        if rem_cell in idx_comp:
            idx_comp.remove(rem_cell)
    cnm.estimates.idx_components = idx_comp
    
    f_remove_components(cnm, rem_cells)
#%%

def f_merge_components2(images, cnm):
    A_corr_thresh = .3
    trace_corr_thresh = .4
    method_wa = 1; # 1 for weighted ave; 0 for full reconstruction corr
    
    A = cnm.estimates.A
    C = cnm.estimates.C
    YrA = cnm.estimates.YrA
    
    dims = cnm.dims
    num_cells, T = C.shape;
    
    if hasattr(cnm.estimates, 'r_values') and cnm.estimates.r_values is not None:
        r_values = cnm.estimates.r_values
    else:
        r_values = np.zeros((num_cells))
    
    A_corr = sc.sparse.triu(A.T * A)
    A_corr.setdiag(0)
    A_corr = A_corr.todense()
    
    traces = C + YrA
    traces_corr = 1 - sc.spatial.distance.cdist(traces, traces, metric='correlation')
    np.fill_diagonal(traces_corr, 0);
    traces_corr = np.triu(traces_corr)
    
    double_corr = np.multiply((A_corr > A_corr_thresh),(traces_corr > trace_corr_thresh))
    
    idx1 = np.where(double_corr)
    
    num_match = len(idx1[0]);
    
    A_new = np.zeros((dims[0]*dims[1], num_match))
    C_new = np.zeros((num_match, T))
    YrA_new = np.zeros((num_match, T))
    rem_cells = []
    SNR_new = np.zeros((num_match))
    cnn_new = np.zeros((num_match))
    S_new = np.zeros((num_match, T))
    F_dff_new = np.zeros((num_match, T))
    for n_match in range(num_match):
        n_cell1 = idx1[0][n_match]
        n_cell2 = idx1[1][n_match]
        
        A1f = A[:,n_cell1]
        A2f = A[:,n_cell2]
        A1_sum = np.sum(A1f)
        A2_sum = np.sum(A2f)
        
        trace1 = traces[n_cell1,:]
        trace2 = traces[n_cell2,:]
        trace1_sum = np.sum(trace1)
        trace2_sum = np.sum(trace2)
        
        if method_wa: # # weighted averages; faster
            trace_comb = (trace1*A1_sum + trace2*A2_sum)/(A1_sum + A2_sum)
            A_comb = (A1f*trace1_sum + A2f*trace2_sum)/(trace1_sum + trace2_sum)
            A_comb_norm = np.linalg.norm(A_comb.todense())
            
            A_combn = (A_comb/A_comb_norm).todense()
            trace_comb_n = trace_comb*A_comb_norm*2  # 2 because we are combining
        else:
            A_idx = ((A1f + A2f) > 0)
            
            full1 = np.dot(A1f[A_idx].T,np.reshape(trace1, (1,T)))
            full2 = np.dot(A2f[A_idx].T,np.reshape(trace2, (1,T)))
            
            full_comb = full1 + full2
            full_trace1 = np.sum(full_comb, axis=0)
            full_trace1n = full_trace1/np.linalg.norm(full_trace1)
            
            full_A = np.dot(full_comb,np.reshape(full_trace1n,(T,1)))
            full_A_norm = np.linalg.norm(full_A)
            full_An = (full_A/full_A_norm)
            
            A_combn = np.zeros((dims[0]*dims[1],1))
            A_combn[np.asarray(A_idx.todense()).flatten()] = full_An
            
            trace_comb_n = np.asarray(full_trace1n*full_A_norm).reshape((T))
        
        
        A_new[:,n_match] = A_combn.flatten()
        C_new[n_match,:] = trace_comb_n
        rem_cells.append(n_cell1)
        rem_cells.append(n_cell2)
        
        A1_2d = np.reshape(A1f.todense(),(dims[0], dims[1]))
        A2_2d = np.reshape(A2f.todense(),(dims[0], dims[1]))
        A_combn_2d = np.reshape(A_combn,(dims[0], dims[1]))
        
        idx2 = np.where(A_combn_2d)
        
        A_cat = np.concatenate((A1_2d, A2_2d, A_combn_2d))
        c_lim = [np.min(A_cat), np.max(A_cat)]
        
        y_lim = [np.min(idx2[0]), np.max(idx2[0])];
        x_lim = [np.min(idx2[1]), np.max(idx2[1])];
        
        if 1: # plot_stuff
            fig = plt.figure()
            gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1])
            ax1 = plt.subplot(gs[0])
            ax1.imshow(A1_2d, vmin=c_lim[0], vmax=c_lim[1]);ax1.set_xlim(x_lim); ax1.set_ylim(y_lim);
            ax1.set_title('cell%d; r_val=%.2f' % (n_cell1, r_values[n_cell1]))
            ax2 = plt.subplot(gs[1])
            ax2.imshow(A2_2d, vmin=c_lim[0], vmax=c_lim[1]); ax2.set_xlim(x_lim); ax2.set_ylim(y_lim)
            ax2.set_title('cell%d; r_val=%.2f' % (n_cell2, r_values[n_cell2]))
            ax3 = plt.subplot(gs[2])
            ax3.imshow(A_combn_2d, vmin=c_lim[0], vmax=c_lim[1]); ax3.set_xlim(x_lim); ax3.set_ylim(y_lim)
            ax3.set_title('A_corr=%.2f; trace_corr=%.2f' % (A_corr[n_cell1, n_cell2], traces_corr[n_cell1, n_cell2]))
            ax4 = plt.subplot(gs[3:6])
            ax4.plot(trace1, linewidth=.5);
            ax4.plot(trace2, linewidth=.5);
            ax4.plot(trace_comb_n, linewidth=.5, color='k')
            ax4.set_title('Traces')
            ax4.autoscale(enable=True, axis='x', tight=True)
            ax4.legend(['cell1', 'cell2', 'comb'])
            fig.suptitle('merging components; method%d' % method_wa)
    
    C2 = np.delete(C, rem_cells, axis=0)
    C3 = np.vstack((C2, C_new))
    
    A1 = np.asarray(A.todense())
    A2 = np.delete(A1, rem_cells, axis=1)
    A3 = np.hstack((A2, A_new))
    
    YrA2 = np.delete(YrA, rem_cells, axis=0)
    YrA3 = np.vstack((YrA2, YrA_new))
    
    cnm.estimates.A = sc.sparse.csr_matrix(A3)
    cnm.estimates.C = C3
    cnm.estimates.YrA = YrA3
    cnm.estimates.idx_components = None
    cnm.estimates.idx_components_bad = None
    
    if hasattr(cnm.estimates, 'r_values'):
        if cnm.estimates.r_values is not None:
            r_values2 = np.delete(r_values, rem_cells)
            r_values3 = np.hstack((r_values2, np.zeros((num_match))))
            cnm.estimates.r_values = r_values3;
        
    if hasattr(cnm.estimates, 'SNR_comp'):
        if cnm.estimates.SNR_comp is not None:
            SNR1 = np.asarray(cnm.estimates.SNR_comp)
            SNR2 = np.delete(SNR1, rem_cells)
            SNR3 = np.hstack((SNR2, SNR_new))
            cnm.estimates.SNR_comp = SNR3;
    
    if hasattr(cnm.estimates, 'cnn_preds'):
        if cnm.estimates.cnn_preds is not None:
            cnn_preds1 = np.asarray(cnm.estimates.cnn_preds)
            cnn_preds2 = np.delete(cnn_preds1, rem_cells)
            cnn_preds3 = np.hstack((cnn_preds2, cnn_new))
            cnm.estimates.cnn_preds = cnn_preds3;
        
    if hasattr(cnm.estimates, 'R'):
        if cnm.estimates.R is not None:
            cnm.estimates.R = cnm.estimates.YrA
        
    if hasattr(cnm.estimates, 'S'):
        if cnm.estimates.S is not None:
            S1 = np.asarray(cnm.estimates.S)
            S2 = np.delete(S1, rem_cells, axis=0)
            S3 = np.vstack((S2, S_new))
            cnm.estimates.S = S3;
    
    if hasattr(cnm.estimates, 'F_dff'):
        if cnm.estimates.F_dff is not None:
            F_dff1 = np.asarray(cnm.estimates.F_dff)
            F_dff2 = np.delete(F_dff1, rem_cells, axis=0)
            F_dff3 = np.vstack((F_dff2, F_dff_new))
            cnm.estimates.F_dff = F_dff3;
              
    if hasattr(cnm.estimates, 'g'):
        if cnm.estimates.g is not None:
            g1 = np.asarray(cnm.estimates.g)
            g2 = np.delete(g1, rem_cells)
            g3 = g2
            for n_m in range(num_match):
                g3 = np.hstack((g3, 1))
                g3[-1] = []
            cnm.estimates.g = g3;
    
    # if hasattr(cnm.estimates, 'bl'):
    #     if cnm.estimates.bl is not None:
    #         cnm.estimates.R = cnm.estimates.YrA
    
    
    # ['S', 'F_dff', 'SNR_comp', 'cnn_preds', 'g', 'bl', 'c1', 'neurons_sn', 'lam']
    
    

        