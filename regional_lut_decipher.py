# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 14:00:27 2021

@author: ys2605
"""

#%%
import numpy as np
import matplotlib.pyplot as plt

#%%

fpath = 'C:\\Users\\ys2605\\Desktop\\stuff\\SLM_GUI\\SLM_control_GUI\\SLM_calibration\\lut_calibration';
fname = 'SLM_3329_20150303.txt'

fname1 = fpath + '\\' + fname;


#%%
file1 = open(fname1, 'rb') 
Lines = file1.readlines()

num_vals = 0
nums_all = [];
for n in range(len(Lines)):
    if n > 15:
        #if n < 40:
        print(str(len(Lines[n])) + ' lines')
        num_vals = num_vals + len(Lines[n])
        for byte in Lines[n]:
            #print(byte)
            nums_all.append(byte)
           
            
nums_all1 = np.asarray(nums_all, order='F')         
    
nums_all2 = np.reshape(nums_all1, [256, 34*103], order='F')

nums_all3 = np.reshape(nums_all2, [256, 103, 34], order='F')

nums_all4 = np.reshape(nums_all3, [103,34], order='F')

nums_all5 = np.reshape(nums_all4[:102,:], [6, 17*17*2], order='F')

#%%


plt.figure();
plt.imshow(nums_all3[:,:,0])
plt.colorbar()

#%%

plt.figure()
plt.plot(nums_all3[0,:,:])
plt.plot(nums_all3[2,:,:])
plt.plot(nums_all3[4,:,:])
plt.plot(nums_all3[6,:,:])
#%%

plt.figure()
plt.plot(nums_all3[:,0,0])


#%%

for byte in Lines[16]:
    print(byte)


#%%
# Using readlines()
file1 = open(fname1, 'rb') 
#  encoding="Western (Windows 1252)" 'cp1252'
#  encoding="Latin-1" 
# , 'rb' read binary

for n in range(18):
    Line = file1.readline()
    if n > 15:
        print(Line)
        for byte in Line:
            print(byte)




 
count = 0
# Strips the newline character

print(Lines[16])

c = 0
for line in Lines[16:]:
    c = c + len(line)


#%%
file = open(fname1, "rb")


for n in range(500):
    byte = file.read(1)
    if n > 300:
        print(byte[0]);
    
    byte.decode('utf-8')