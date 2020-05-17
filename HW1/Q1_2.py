#!/usr/bin/env python
# coding: utf-8

# In[224]:


import numpy as np
import matplotlib.image as mpimg
import cv2
import time
from multiprocessing import Pool
from cacfunc import *
from PIL import Image
from scipy import ndimage
from scipy.ndimage import filters
from pylab import *
from skimage import color, transform, io
import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial
from matplotlib.patches import Circle
from scipy.ndimage.filters import gaussian_laplace, convolve
import time
import math
from itertools import chain
from scipy.ndimage.filters import rank_filter, generic_filter
from scipy.misc import imresize


# In[132]:


def load_and_preprocess(path):
    # Load image.
    img = cv2.imread(str(path),0)
    #how to conv to greyscale lol
    img = img/255.0
    #normalize intensity
    return img


# In[222]:


# Get squared Laplacian response in scale-space.
def get_scale_space(image, init_sigma, levels, k,method='downsample'):
    h,w = image.shape[0],image.shape[1]
    
    sigma = np.zeros(levels)
#     sigma = np.asarray([0]*levels)
    sigma[0] = init_sigma
    
    scale_space = np.zeros((h, w, levels))
    ord=3
    # Method 2. (faster version)
    
    start = time.time()
    # Ensure odd filter size.
    n=np.ceil(sigma*9)
    filter_size = int(n[0])

    
    if filter_size % 2 != 0:
        filter_size = int(filter_size)
    else:
        filter_size = filter_size+ 1

    # Initialize filter matrix.

    gauss_filter = np.zeros((filter_size, filter_size))
    gauss_filter[(( 1+filter_size )//2 - 1)//1 ][(( 1+filter_size )//2 - 1)//1 ] = 1
#     gauss_filter[center][center] = 1


    # Obtain filter (no normalization needed).
    LoG = gaussian_laplace(gauss_filter, init_sigma)

    # Scale the image.
    for i in range(0,levels,1):
        # Down scale.
        scaled_h = ( ((1//k) ** i) * h)//1
        scaled_w = ( ((1//k) ** i) * w)//1
#             scaled_im = transform.resize(image, (scaled_h, scaled_w), order=3)
        scaled_im = transform.rescale(image, (1/k) ** i, order=ord)

        # Apply convolution without normalization.
        im_tmp = convolve(scaled_im, LoG) ** (ord-1)

        # Upscale.
        scale_space[:,:,i] = transform.resize(im_tmp, (h, w), order=ord)

        # Update sigma.
        if (i+1 < levels):
            sigma[i+1] = sigma[i] * k

   


    return scale_space, sigma


# In[215]:


# Helper func: non-maximum suppression in each 2D slice.
def non_max_sup_2D(scale_space,method='rank'):
    h,w = scale_space.shape[0],scale_space.shape[1]
    
    levels = scale_space.shape[2]
    
    local_max = np.empty((h, w, levels))
    
    # Method 1: rank_filter.

    start = time.time()
    for i in range(levels):
        curr_response = scale_space[:,:,i]
        local_max[:,:,i] = rank_filter(curr_response, -1, (3,3))

    
    return local_max


# In[274]:


# Non-maximum suppression in 3D scale space.
def non_max_sup_3D(scale_space, sigma):
    
    # Obtain local 2D non max sup using rank_filter.
    
    h,w = non_max_sup_2D(scale_space, 'rank').shape[0],non_max_sup_2D(scale_space, 'rank').shape[1]
    levels = non_max_sup_2D(scale_space, 'rank').shape[2]
    
    # Compute non-max suppression accorss all layers.
    global_max=nonmaxlayers(local_max,h,w)
    
    global_max=dup(levels,global_max,scale_space,[],[],[],False)

    return global_max


# In[136]:


# Helper func: compute radius of each local maximum.
def get_radius(sigma, num_rads): 
    return ((np.zeros(num_rads)+1)  * sigma * 1.414)


# In[159]:


# Helper func: mask filter to eliminate boundaries noises.
def get_mask(h, w,  sigma, levels):
    mask = np.zeros((h, w, levels))
    for i in range(0,levels,1):
        b = (math.ceil( 1.414 * sigma[i] ))//1  # Boundary.
        mask[b+1:h-b, b+1:w-b] = 1.0
    return mask


# In[145]:


def nonmaxlayers(local_max,h,w):
    global_max = np.zeros(local_max.shape)
    for i in range(h):
        for j in range(w):
            max_value = np.amax(local_max[i,j,:])
            max_idx = np.argmax(local_max[i,j,:])
            global_max[i,j,max_idx] = max_value
    return global_max


# In[244]:


def dup(levels,global_max,scale_space,threshold,mask,sigma,blob):
    if not blob:
        # Eliminate duplicate values.
        for i in range(0,levels,1):
            global_max[:,:,i] =  np.asarray(np.where( (  scale_space[:,:,i] == global_max[:,:,i] ), global_max[:,:,i], 0))
        return global_max
    
    row_idx,radius,col_idx = [],[],[]
    temp=[]
    for i in range(0,levels,1):
        global_max[:,:,i] = np.asarray(np.where( (mask[:,:,i] == 1) & (global_max[:,:,i] > threshold) , 1, 0))
        # Obtain row & column index for local maxima
        row_idx.append( list( np.where(global_max[:,:,i] == 1)[0] ) )
        col_idx.append( list( np.where(global_max[:,:,i] == 1)[1] ) )
        
        temp.append( (i,sigma) )
        # Compute radius.
        radius.append(list(get_radius(sigma[i], len(row_idx[i]))))
    # Flatten nested list.
    radius = list(chain.from_iterable(radius))
    row_idx = list(chain.from_iterable(row_idx))
    col_idx = list(chain.from_iterable(col_idx))
    return global_max,col_idx,row_idx,radius


# In[198]:


# Obtain center points and radius of blobs.
def detect_blob(global_max, threshold, sigma):
    levels = global_max.shape[2]
    
    mask = get_mask(global_max.shape[0], global_max.shape[1], sigma, levels)
    
    row_idx,radius,col_idx = [],[],[]

    global_max,col_idx,row_idx,radius=dup(levels,global_max,[],threshold,mask,sigma,True)
    

    print('Done with finding blobs.')
    return row_idx, col_idx, radius


# In[266]:


def show_and_save_all_circles(image, cx, cy, rad, 
                              root, img_name, method, threshold, color='yellow'):

    
    data=np.asarray( list(zip(cx,cy,rad))  )
    img_name=img_name.split(".")
    img_name=img_name[0]
    
    np.save('keypoints/'+str(img_name), data)


# In[260]:


# Pipeline for running the whole program.
def run_detection(root, img_name, method, levels, k, threshold, init_sigma):
    # Load and preprocess image.
    path = str(root+img_name)
    img = load_and_preprocess(path)
    
    # Get response.
    scale_space, sigma = get_scale_space(img, init_sigma, levels, k, method)
    
    # Non-max suppression.
    global_max = non_max_sup_3D(scale_space, sigma)
    
    # Get blobs.
    row_idx, col_idx, radius = detect_blob(global_max, threshold, sigma)
    
    # Display and save output.
    show_and_save_all_circles(img, col_idx, row_idx, radius, 
                              root, img_name, method, threshold)


# In[271]:


root = 'D:/GitHub/Semester-6/Multimedia Computing and Application/Homeworks/HW1/'
k = 1.2  # Scale factor
# Choose one image.
images = ['9ball.jpg']
# Choose one method.
init_sigma = 2 
# methods = [ 'downsample']

# Configure parameters.
levels = 11  # Scale levels



# Good thresholds for method 0 and 1.
thresholds = np.array([0.0000001, 0.0019,0.003])


# In[272]:


run_detection(root, images[0], 'downsample', levels, k, thresholds[1], init_sigma)


# In[258]:


imagelist=np.asarray([])

path = root+str('images/')
for images in os.listdir(path):
    imagelist = np.append(imagelist, images)


# In[268]:


imagelist


# In[273]:


for i in imagelist:
    run_detection(path, i, 'downsample', levels, k, thresholds[1], init_sigma)

