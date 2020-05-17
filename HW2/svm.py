#!/usr/bin/env python
# coding: utf-8

# In[102]:


import numpy as np
import cmath
from scipy.fftpack import dct
from scipy.fftpack import fft
from matplotlib import pyplot as plt
import librosa as lr
from scipy.cluster.vq import vq, kmeans, whiten
import sys
np.set_printoptions(threshold=sys.maxsize)
import IPython.display as ipd
import math
import librosa.display
from scipy.signal import get_window
import os
import warnings
warnings.filterwarnings("ignore")


# In[128]:


x=[]
y=[]

path='zero'
o=0
limit=5
for i in os.listdir(path+'/'):
    
    i=i+''
    if(i!='.ipynb_checkpoints' and i!=path+'.ipynb' and i!='spec'):
        
        o+=1
        temp=np.load(path+'/spec/'+i+'.npy').flatten()
        temp[temp == np.inf] = 0
        if(len(temp)==61440):
            x.append(list(temp) )
            y.append(0)
    if(o==limit):break

path='one'
o=0
for i in os.listdir(path+'/'):
    
    i=i+''
    if(i!='.ipynb_checkpoints' and i!=path+'.ipynb' and i!='spec'):
        o+=1
        temp=np.load(path+'/spec/'+i+'.npy').flatten()
        temp[temp == np.inf] = 0
        if(len(temp)==61440):
            x.append(list(temp) )
            y.append(1)
    if(o==limit):break
        
path='two'
o=0
for i in os.listdir(path+'/'):
    
    i=i+''
    if(i!='.ipynb_checkpoints' and i!=path+'.ipynb' and i!='spec'):
        o+=1
        temp=np.load(path+'/spec/'+i+'.npy').flatten()
        temp[temp == np.inf] = 0
        if(len(temp)==61440):
            x.append(list(temp) )
            y.append(2)
    if(o==limit):break

path='three'
o=0
for i in os.listdir(path+'/'):
    
    i=i+''
    if(i!='.ipynb_checkpoints' and i!=path+'.ipynb' and i!='spec'):
        o+=1
        temp=np.load(path+'/spec/'+i+'.npy').flatten()
        temp[temp == np.inf] = 0
        if(len(temp)==61440):
            x.append(list(temp) )
            y.append(3)
    if(o==limit):break

path='four'
o=0
for i in os.listdir(path+'/'):
    
    i=i+''
    if(i!='.ipynb_checkpoints' and i!=path+'.ipynb' and i!='spec'):
        o+=1
        temp=np.load(path+'/spec/'+i+'.npy').flatten()
        temp[temp == np.inf] = 0
        if(len(temp)==61440):
            x.append(list(temp) )
            y.append(4)
    if(o==limit):break
        
path='five'
o=0
for i in os.listdir(path+'/'):
    
    i=i+''
    if(i!='.ipynb_checkpoints' and i!=path+'.ipynb' and i!='spec'):
        o+=1
        temp=np.load(path+'/spec/'+i+'.npy').flatten()
        temp[temp == np.inf] = 0
        if(len(temp)==61440):
            x.append(list(temp) )
            y.append(5)
    if(o==limit):break

path='six'
o=0
for i in os.listdir(path+'/'):
    
    i=i+''
    if(i!='.ipynb_checkpoints' and i!=path+'.ipynb' and i!='spec'):
        o+=1
        temp=np.load(path+'/spec/'+i+'.npy').flatten()
        temp[temp == np.inf] = 0
        if(len(temp)==61440):
            x.append(list(temp) )
            y.append(6)
    if(o==limit):break
        
path='seven'
o=0
for i in os.listdir(path+'/'):
    
    i=i+''
    if(i!='.ipynb_checkpoints' and i!=path+'.ipynb' and i!='spec'):
        o+=1
        temp=np.load(path+'/spec/'+i+'.npy').flatten()
        temp[temp == np.inf] = 0
        if(len(temp)==61440):
            x.append(list(temp) )
            y.append(7)
    if(o==limit):break
        
path='eight'
o=0
for i in os.listdir(path+'/'):
    
    i=i+''
    if(i!='.ipynb_checkpoints' and i!=path+'.ipynb' and i!='spec'):
        o+=1
        temp=np.load(path+'/spec/'+i+'.npy').flatten()
        temp[temp == np.inf] = 0
        if(len(temp)==61440):
            x.append(list(temp) )
            y.append(8)
    if(o==limit):break
        
path='nine'
o=0
for i in os.listdir(path+'/'):
    
    i=i+''
    if(i!='.ipynb_checkpoints' and i!=path+'.ipynb' and i!='spec'):
        o+=1
        temp=np.load(path+'/spec/'+i+'.npy').flatten()
        temp[temp == np.inf] = 0
        if(len(temp)==61440):
            x.append(list(temp) )
            y.append(9)
    if(o==limit):break


# In[143]:


from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,shuffle=True)
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

clf = SVC(kernel='linear')

clf.fit( x_train , 
        y_train)

y_pred = clf.predict( x_test)
print(accuracy_score(y_test,np.ceil(y_pred)))


# In[80]:


y_train.flatten()


# In[127]:




