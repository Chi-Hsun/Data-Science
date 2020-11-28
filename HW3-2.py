import pandas as pd
import numpy as np
import random
import math

def gen_features(data, status):
    k = 0
    features = np.empty((int(sum(status)),62))
    for i in range(len(data)):
        if status[i]==1:
            features[k] = data[i]
            k = k+1
    return features

def classify(data, label):
    A = 0
    for i in range(len(label)):
        if label[i] == 1:
            A = A+1
    a=0
    b=0
    classA = np.empty((A,2000))
    classB= np.empty((len(label)-A,2000))
    for s in range(len(label)):
        if label[s] ==1:
            classA[a] = data.T[s]
            a = a+1
        else:
            classB[b] = data.T[s]
            b = b+1
    return classA, classB

def obj(data, label):
    A = 0
    B = 0
    for i in range(len(data)):
        A = A + abs(np.corrcoef(data[i],label)[1,0])
    mean_A = A/len(data)    

    for k in range(len(data)):
        for j in range(k, len(data)):
            if k != j:
                B = B + abs(np.corrcoef(data[k],data[j])[1,0])
    mean_B = 2*B/(len(data)*len(data)-len(data))
    return mean_B-mean_A #the smaller the better

def SA(data, label, cooling_rate, T0, max_iteration, disturbance_num):    
    current_status = np.random.uniform(0,2000,2000)
    for i in range(2000):
        if current_status[i]<10:
            current_status[i] = 1
        else:
            current_status[i] = 0
    features = gen_features(data, current_status)        
    fitness_curr = obj(features, label)  
    temp_status = current_status
    best_status = current_status    
    fitness_best = fitness_curr
    T = T0
    
    for i in range(max_iteration):
        disturbance = np.random.randint(0,2000,size=disturbance_num)
        for s in disturbance:
            temp_status[s] = 1-temp_status[s]
            
        features = gen_features(data, temp_status)
        fitness_temp = obj(features, label) 
        delta = fitness_temp - fitness_curr
        
        if delta<0:
            fitness_curr = fitness_temp
            current_status = temp_status
        else:
            prob = random.random()
            if prob<math.exp(-delta/T):
                fitness_curr = fitness_temp
                current_status = temp_status
                
        if fitness_curr<fitness_best:
            fitness_best = fitness_curr
            best_status = current_status
        T = T*cooling_rate
    output = []    
    for i in range (2000):
        if best_status[i] == 1:
            output.append(i)
    print('Select' ,len(output), 'features')
    return output

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
        
a = SA(genes, tumor, 0.5, 10000, 10, 1)
print(a)
