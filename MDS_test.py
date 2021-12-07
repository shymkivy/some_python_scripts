# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 13:18:03 2021

MDS example
converd high D data into some distance space, such as eucleidian or cosine
then try to estimate low D data that creates similar distance matrix
essentially trying to estimate High D covariance in low D structure.


@author: ys2605
"""




#%%

from sklearn import datasets
import math as ma
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

#%%

digits = datasets.load_digits(n_class = 6)

data = digits.data


#%%
plt.figure;
plt.imshow(np.reshape(data[500,:], [8, 8]))
plt.show();


#%%
num_im = np.shape(data)[0]

#%% 
def eucl_dist(mat1, mat2):
    # both inputs matrix (measurements X features)
    d1 = np.shape(mat1)[0]
    d2 = np.shape(mat2)[0]
    dist1 = np.zeros((d1,d2))
    for n_im in range(d1):
        dist1[n_im,:] = np.sqrt(np.sum((mat1[n_im,:] - mat2)**2,axis=1));
    return dist1;

#%% compute distances between instances of data

dist_data = eucl_dist(data, data)

# sort images by targes te get clear plots of results
idx1 = np.argsort(digits.target)

dist_data_sort = eucl_dist(data[idx1,:], data[idx1,:])


#%% initialize random x,y locations for each image

n,p = data.shape
dim = 2
X = np.random.rand(n,dim)

#%%
def compute_B(d_data, d_X):
    d_data,d_X = np.array(d_data), np.array(d_X)
    d_X[d_X==0.0] = np.inf
    z = -d_data/d_X
    z_diag = -(np.sum(z, axis=1)) #  - np.diag(z)
    np.fill_diagonal(z, z_diag);
    return z


#%%
sigma_old = np.inf
tol_thresh = 1e-4
max_iter = 10000

#%% run algorithm
tol1 = 99999
ii = 0
while (ii<max_iter and tol1>tol_thresh):
    # compute distances of current X
    dist_X = eucl_dist(X, X);
    # error (sigma) between x covariances and real data covariances
    sigma_new = sigma = np.sum((dist_data - dist_X)**2);
    tol1 = sigma_old - sigma_new;
    print('iter ' +  str(ii)  +  '; sigma old: ' + "{:.2e}".format(sigma_old) + '; sigma new: '  + "{:.2e}".format(sigma_new) + '; tol = '+ "{:.2e}".format(tol1))
    sigma_old = sigma_new
    # compute matrix B
    B = compute_B(dist_data, dist_X)
    # multiply b * X
    XB = np.dot(X.T,B).T/n # here I am not sure why this is divided by n
    X = XB - np.mean(XB, axis=0); # maybe not necessary because the mat is zeroed already
    ii = ii + 1;

#%%

dist_X_sorted = eucl_dist(X[idx1,:], X[idx1,:]);


#%%
plt.figure()
plt.subplot(221)
plt.imshow(dist_data)
plt.title('im euc distances')
plt.subplot(222)
plt.imshow(dist_data_sort)
plt.title('im dist sorted')
plt.subplot(223)
plt.imshow(dist_X)
plt.title('X sorted pre')
plt.subplot(224)
plt.imshow(dist_X_sorted)
plt.title('X sorted post')
plt.show()


colors1 = list(mcolors.TABLEAU_COLORS.values())
plt.figure();
colors = []
for n_num in digits.target_names:
    if np.sum(digits.target==n_num):
        plt.plot(X[digits.target==n_num,0],X[digits.target==n_num,1], 'o', color=colors1[n_num%len(colors1)], label=str(n_num));
plt.legend()
plt.title('Embeded im distances into 2D')
plt.show()