#!/usr/bin/env python
# coding: utf-8

# In[141]:


import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import IPython.display as ipd
from scipy.fftpack import dct
import warnings
import librosa.display
from scipy.signal import get_window
import cmath
import os
from scipy.cluster.vq import vq, kmeans, whiten
import sys
import pandas as pd
warnings.filterwarnings("ignore")
import math
from scipy.fftpack import fft
from matplotlib import pyplot as plt
import librosa as lr


# In[142]:


def inputtooutput(audio):
    y,_1=lr.load(audio)
    pos=math.ceil(np.log2(len(y)))
    outlength=pow(2,pos)
    if( not (outlength==len(y)) ):
        outaud=np.pad(y,(0,outlength-len(y)),'constant')    
    return outaud,_1

lis=[]
# os.mkdir('spec')
for i in os.listdir():
    i=i+''
    if(i!='.ipynb_checkpoints' and i!='zero.ipynb' and i!='spec' and i!='Untitled.ipynb'):
        lis.append(i)
len(lis)


# In[143]:


audio=str(lis[1]) #change this name to give any other output
outaud,fs=inputtooutput(audio)


# In[144]:


def FFT_own1(x):
    x = np.asarray(x, dtype=float)
    
    n = np.arange(x.size)
    if len(x) & 1 == 1:
        raise ValueError("Error in dims")
    
    elif len(x) < 33:
        n = np.arange(x.size)
        k = n.reshape(x.size, 1)
        x = np.asarray(x, dtype=float)
        
        M = np.exp((-1)* k * n * (2j) * np.pi / x.size)
        ret = np.dot(M, x)
        return ret
    
    elif True:
        odd = FFT_own1(x[1::2])
        even = FFT_own1(x[0::2])
        angle = np.exp((-1)* n * (2j) * np.pi / x.size)
        arr=[ odd * angle[:x.size//2 ] +even    ,    odd*angle[x.size//2:] +even ]
        ret=np.concatenate(arr)
        return ret


# In[147]:


fig=plt.figure(figsize=(15,4))
plt.subplot(1,1,2)
plt.plot(abs(FFT_own1(outaud)))


# ![image.png](attachment:image.png)

# In[151]:


def own_specgram(y,n_fft,noverlap = None):
    ffts = []
    noverlap = (n_fft//2)//1
    x=n_fft-noverlap
    temp  = np.arange(0,y.size,x,dtype=int)
    s=temp + n_fft
    windows  = temp[s < y.size]
    for x in windows:
        start=x
        final=x+n_fft
        audioframed=y[start:final]
        window = get_window("hanning", n_fft)
        final=window*audioframed
        ffts.append(FFT_own1(final) )
    spec = np.log(np.array(ffts).T)
    return spec

spec=own_specgram(y=outaud,n_fft=2048)


plt.figure(figsize=(15,4))
plt.subplot(1,2,2)
plt.xlabel('All Windows')
plt.ylabel('Frame Length')
plt.imshow((abs(spec)),  aspect="auto", cmap="viridis_r")
plt.title('Spectogram')




# In[77]:
for i in lis:
    audio=i 
    outaud,fs=inputtooutput(audio)
    fftout=FFT_own1(outaud)
    _1,spec,_2=own_specgram(y=outaud,n_fft=L)
    powarr=(np.square(abs((np.asarray(ffts2)))))  
    np.save('spec/'+i,absspec)


# In[ ]:




