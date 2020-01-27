import numpy as np

A = np.array([[1,2,3],[4,5,6]])

print(A)

B = np.array([[7,8]])

A = np.concatenate((A,B.T),axis=1)

print(A)
