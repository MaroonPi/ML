import numpy as np
w = np.random.normal(0,0.1,(3,5))
b = np.random.normal(0,0.1,(1,5))

x = np.random.normal(0,0.1,(6,3))

print(np.dot(x,w)+b)

grad = np.array([[6,5,3,2,4],[1,5,1,2,1],[8,5,9,2,5],[1,5,1,2,5],[9,7,3,8,5],[2,9,5,2,1]])

gradb = np.sum(grad,axis=0)

print(gradb)

d = np.array([2,3,0,1])
mask = d > 0
mask = mask.astype(int)
print(mask)
