import t_sne_bhcuda.bhtsne_cuda as tsne_bhcuda
import matplotlib.pyplot as plt
import numpy as np
import time
import scipy.io as sio
from sklearn.decomposition import PCA
import os


#os.environ["CUDA_VISIBLE_DEVICES"]="1"

perplexity = 30.0
theta = 0.5
learning_rate = 500.0
iterations = 1000
gpu_mem = 0.8
samples = 0

#data_for_tsne = np.random.rand(180792,2048)
t1=time.time()
mylist=[]

with open('/home/scopeserver/RaidDisk/DeepLearning/mwang/tensorflow/tensorflow/models/image/imagenet/parts_dresses_feature.txt','r') as f:
    #for line in f.readlines():
        #mylist.append(line)
    mylist=f.readlines()

feature_v3=np.asarray(mylist,dtype=float)
print (feature_v3.shape)
print (feature_v3[0:10])
delta=time.time()-t1
print (delta)

feature_v3 = np.reshape(feature_v3,(int(len(feature_v3)/2048),2048))
print (feature_v3[0,0:10])

print (feature_v3.shape)

#sio.savemat('footwear.mat',{'footwear':feature_v3})

p = PCA(n_components=768)
p.fit(feature_v3)
after_pca = p.fit_transform(feature_v3)
print(after_pca[0,0:10])
print (after_pca.shape)
#sio.savemat('footwear_afterPCA3.mat',{'footwear_afterPCA':after_pca})

t_sne_result = tsne_bhcuda.t_sne(samples=after_pca, files_dir="./",
                        no_dims=3, perplexity=perplexity, eta=learning_rate, theta=theta,
                        iterations=iterations,seed=samples, gpu_mem=gpu_mem, randseed=-1, verbose=2)
#t_sne_result = np.transpose(t_sne_result)#

sio.savemat('dresses_newxy3.mat',{'Y':t_sne_result})

#np.savetxt('./tsne_xy.txt',t_sne_result)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(t_sne_result[0], t_sne_result[1])
#fig.show()
