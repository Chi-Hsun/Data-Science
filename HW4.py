import numpy as np
from numpy import linalg as LA

class1 = np.mat([[5,3,3,4,5], [3,5,4,7,6]])
class2 = np.mat([[9,7,8,7,10], [10,7,5,2,8]])

# mean for each class
mean_1 = np.mean(class1, axis=1)
mean_2 = np.mean(class2, axis=1)

class1_sub_mean = class1 - mean_1


# calculate SW 
S1 = np.zeros([2,2])
S1 = np.matmul(class1_sub_mean, class1_sub_mean.transpose())
class2_sub_mean = class2 - mean_2
S2 = np.zeros([2,2])
S2 = np.matmul(class2_sub_mean, class2_sub_mean.transpose())
SW = S1+S2
print('S1 = ', S1)
print('S2 = ', S2)
print('SW = ', SW)
# calculate SB
SB = (mean_1 - mean_2)*((mean_1 - mean_2).transpose())

# eigenvalue & eigenvector
eigenvalue, eigenvector = LA.eig(LA.inv(SW)*SB)
print("eigenvalue = ", eigenvalue)
print("eigenvector = ", eigenvector)

optimal_projection_vector = eigenvector[:,0]
print("optimal_projection_vector = ", optimal_projection_vector)
