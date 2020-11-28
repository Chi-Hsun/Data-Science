import pandas as pd
import numpy as np

def T_statistics(class1,class2):
    class1_mean = class1.mean(axis=0)
    class2_mean = class2.mean(axis=0)
    class1_std = class1.std(axis=0)
    class2_std = class2.std(axis=0)
    tvalue = abs(class1_mean-class2_mean)/np.sqrt(class1_std*class1_std/len(class1)+class2_std*class2_std/len(class2))
    return tvalue


path = "/Users/betty92218/Downloads/hw3_Data"
genes = np.genfromtxt(path+'/gene.txt')
label = np.genfromtxt(path+'/label.txt')
tumor = np.array(label)
tumor_num = 0;
for i in range(len(label)):
    if int(label[i])>0:
        tumor[i] = 1
    else:
        tumor_num = tumor_num+1
        tumor[i] = -1
#classify the data
a=0
b=0
notumor_data = np.empty((62-tumor_num,2000))
tumor_data = np.empty((tumor_num,2000))
for i in range(len(tumor)):
    if tumor[i] ==1:
        notumor_data[a] = genes.T[i]
        a = a+1
    else:
        tumor_data[b] = genes.T[i]
        b = b+1
#t-statistics

tvalues = T_statistics(tumor_data,notumor_data)
selected = (-tvalues).argsort()[0:3]
print(selected)
