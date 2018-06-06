#import t_sne_bhcuda.bhtsne_cuda as tsne_bhcuda
#import matplotlib.pyplot as plt
import numpy as np
import time
import scipy.io as sio
#from sklearn.decomposition import PCA
import os
import h5py
import Image



with h5py.File('high_res_image.mat', 'r') as file:
    image = file['G'].value.transpose()

img = Image.fromarray(image)
img.save('high_res.jpg')
