# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 17:00:03 2021

@author: ys2605
"""

#%%

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from scipy import signal

from pyspark.mllib.linalg.distributed import IndexedRowMatrix, IndexedRow, BlockMatrix
from pyspark.mllib.feature import StandardScaler
from pyspark.mllib.linalg import Vectors, DenseMatrix, Matrix
from sklearn import datasets
from pyspark import SparkContext
from pyspark.sql import SparkSession

#%%


U1 = np.random.uniform(-1, 1, 1000)
U2 = np.random.uniform(-1, 1, 1000)

G1 = np.random.randn(1000)
G2 = np.random.randn(1000)



#%%
fig = plt.figure();

ax1 = fig.add_subplot(121, aspect = "equal")

ax1.scatter(U1, U2, marker = ".")

ax1.set_title("Uniform")

ax2 = fig.add_subplot(122, aspect = "equal")
ax2.scatter(G1, G2, marker = ".")
ax2.set_title("Gaussian")

plt.show()


#%%

A = np.array([[1, 0], [1, 2]])

U_source = np.array([U1, U2])
U_mix = U_source.T.dot(A)

G_source = np.array([G1, G2]);
G_mix = G_source.T.dot(A)


fig  = plt.figure()

ax1 = fig.add_subplot(121)
ax1.set_title("Mixed Uniform ")
ax1.scatter(U_mix[:, 0], U_mix[:,1], marker = ".")

ax2 = fig.add_subplot(122)
ax2.set_title("Mixed Gaussian ")
ax2.scatter(G_mix[:, 0], G_mix[:, 1], marker = ".")


plt.show()   


#%%

U_pca = PCA(whiten=False).fit_transform(U_mix)


G_pca = PCA(whiten=False).fit_transform(G_mix)

fig  = plt.figure()

ax1 = fig.add_subplot(121)
ax1.set_title("PCA Uniform ")
ax1.scatter(U_pca[:, 0], U_pca[:,1], marker = ".")

ax2 = fig.add_subplot(122)
ax2.set_title("PCA Gaussian ")
ax2.scatter(G_pca[:, 0], G_pca[:, 1], marker = ".")



#%%

# same as applying fit_transform
U_pca2 = PCA(whiten=True).fit(U_mix)
U_tr = (U_mix - U_pca2.mean_.T).dot(U_pca2.components_)# + U_pca2.mean_.T

fig  = plt.figure()

ax1 = fig.add_subplot(121)
ax1.set_title("PCA Uniform ")
ax1.scatter(U_tr[:, 0], U_tr[:,1], marker = ".")


#%% do ICA

np.random.seed(0)
n_samples = 3000
t = np.linspace(0,10, n_samples)

# create signals sources
s1 = np.sin(3*t) # a sine wave
s2 = np.sign(np.cos(6*t)) # a square wave
s3 = signal.sawtooth(2 *t) # a sawtooth wave


# combine single sources to create a numpy matrix
S = np.c_[s1,s2,s3]

S += 0.2*np.random.normal(size = S.shape)


#%%

# create a mixing matrix A
A = np.array([[1, 1.5, 0.5], [2.5, 1.0, 2.0], [1.0, 0.5, 4.0]])
X = S.dot(A.T)

#%%

plt.figure();
plt.plot(t, X)
plt.show()

#plot the single sources and mixed signals
plt.figure(figsize =(26,12) )
colors = ['red', 'blue', 'orange']

plt.subplot(2,1,1)
plt.title('True Sources')
for color, series in zip(colors, S.T):
    plt.plot(series, color)
plt.subplot(2,1,2)
plt.title('Observations(mixed signal)')
for color, series in zip(colors, X.T):
    plt.plot(series, color)


#%%



spark = SparkSession.builder.appName('SparkByExamples.com').getOrCreate()
sparkContext=spark.sparkContext

X_rdd = sparkContext.parallelize(X).map(lambda X:Vectors.dense(X) )
scaler = StandardScaler(withMean = True, withStd = False).fit(X_rdd)

X_sc = scaler.transform(X_rdd)

#%%
X_pca = PCA(whiten=False).fit_transform(X)

plt.figure();
plt.plot(t, X_pca)
plt.show()
