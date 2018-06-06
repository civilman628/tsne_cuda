import numpy as np
import time

mylist=[]

t1=time.time()
with open('/home/scopeserver/RaidDisk/DeepLearning/mwang/tensorflow/tensorflow/models/image/imagenet/wholeperson_feature.txt','r') as f:
    #for line in f.readlines():
     #   mylist.append(line)
    mylist=f.readlines()

delta=time.time()-t1
print (delta)
list_array=np.asarray(mylist,dtype=float)
print (list_array.shape)
print(len(list_array))