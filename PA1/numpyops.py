import numpy as np
from knn import KNN
from collections import Counter

givenList = [1,1,1,0,0,0]
counter = Counter(givenList)
print(counter.most_common())
if(len(counter.most_common())==2):
    common_label = 0

print(common_label)
