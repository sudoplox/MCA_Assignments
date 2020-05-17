#!/usr/bin/env python
# coding: utf-8

# In[27]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import time
from multiprocessing import Pool
from cacfunc import *
from PIL import Image
from sklearn.cluster import MiniBatchKMeans


# In[ ]:


from PIL import Image
img = Image.open('images/all_souls_000047.jpg',)
# img.save('greyscale.png')
plt.imshow(img)

# gray=gray.resize( (720,1080) )
img1=img.convert( 'L',palette=Image.ADAPTIVE, colors=4*4*4)
#L = R * 299/1000 + G * 587/1000 + B * 114/1000
# img1  #big gray image lol


# In[ ]:


img=img.quantize(4*4*4)
img.convert('L')


# In[ ]:


imgarr= np.asarray(img) 
imgarr1= np.asarray(img1) 


# In[ ]:


imgarr.flatten()


# In[ ]:


d=[ 1, 3, 5, 7]
imgarr.shape


# In[ ]:





# In[28]:


def unique1(a):
    sorted_idx = np.lexsort(np.asarray(a).T)
    sorted_data =  a[sorted_idx,:]
    # Get unique row mask
    row_mask = np.append([True],np.any(np.diff(sorted_data,axis=0),1))

    # Get unique rows
    colors65 = sorted_data[row_mask]
    return colors65


# In[17]:


def unique(a):
    """
    remove duplicates from input list
    """
    a=np.asarray(a)
    order = np.lexsort(a.T)
    a = a[order]
    #The first order difference is given by out[n] = a[n+1] - a[n] along the given axis, higher order differences are calculated by using diff recursively.
    diff = np.diff(a, axis = 0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis = 1)
    return a[ui]


# In[29]:


def isValid(X, Y, point):
    """
    Check if point is a valid pixel
    """
    if point[0] < 0 or point[0] >= X or point[1] < 0 or point[1] >= Y:
        return False
    return True


# In[30]:


def getNeighbors( X,Y, x, y, dist):
    """
    Find pixel neighbors according to various distances
    """
 
    pt1=( x-dist , y-dist ) #bottom left
    pt2=( x-dist , y+dist ) #top left
    pt3=( x+dist , y+dist ) #top right
    pt4=( x+dist , y-dist ) #bottom right
    
    points=[]
    points.append(pt1)
    points.append(pt2)
    points.append(pt3)
    points.append(pt4)
#     points = (cn1, cn2, cn3, cn4, cn5, cn6, cn7, cn8)
    Cn = []
    
    for i in range(x-dist+1,x+dist):  #
        points.append( (i,y+dist) )
        points.append( (i,y-dist) )
    
    for i in range(y-dist+1,y+dist):
        points.append( (x+dist,i) )
        points.append( (x-dist,i) )
    
    for i in points:
        if isValid(X, Y, i):
            Cn.append(i)

    return Cn


# In[31]:


def correlogram(photo, Cm, K):
    print(photo.shape)
    colorsPercent = []
    X, Y, _ = photo.shape
    cou=0
    freq=[0]*len(Cm)
    
    for x in range(0, X): #iterating over x
        for y in range(0, Y):  #iterating over y
            for m in range(len(Cm)):  #available colours for freq array
                if np.array_equal(Cm[m], photo[x][y]):
                    freq[m]+=1    
    for k in K:
#         print ("k: ", k)
#         countColor = 0
        color = []
        for i in Cm:
            color.append(0) 
        
        for x in range(0, X): #iterating over x
            for y in range(0, Y):  #iterating over y
#                 print(cou)
                cou+=1
                Ci = photo[x][y]
                           
                Cn = getNeighbors(X, Y, x, y, k)
                for j in Cn:  #neighbours
                    Cj = photo[j[0]][j[1]]

                    for m in range(len(Cm)):  #available colours
                        if np.array_equal(Cm[m], Cj) and np.array_equal(Cm[m], Ci) :
                            color[m] =+ 1
#                             countColor = countColor + 1
#         for i in range(len(color)):
#             color[i] = float(color[i]) / (countColor)
        
        colorsPercent.append(color)
#         print(color)        
    return colorsPercent,freq


# In[32]:


def cac(imgn,name):
    # according to "Image Indexing Using Color Correlograms" paper
    d = [2]
    
    if name==1:
        pass
    elif name==2:
        img=imgn
        res=img
        colors64= [[0,0,0],[200,200,200]]
        res2=img
    elif name==3:
        img=cv2.imread('64transformed/'+imgn)
        scale_percent = 7 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
#         print(np.asarray(img).shape)
#         print( sorted(np.asarray(img)))
        colors64= unique1(np.asarray(img).reshape(-1, img.shape[-1]) )
#         print(colors64)
        res2=img
#     colors64 = unique(res)
    print(1)
    result = correlogram(res2, colors64, d)
    print("done!")
    np.save( str('CACsPart2/'+imgn), np.asarray(result))
    return result


# In[33]:


imagelist=np.asarray([])
imagelist1=np.asarray([])

path = "64transformed/"
for images in os.listdir(path):
    imagelist = np.append(imagelist, images)
#     imagelist = np.append(imagelist, cv2.imread(str('images/'+images), 1))
#     imagelist.append(cv2.imread(str('images/'+images), 1))
for images in os.listdir("images/"):
    imagelist1 = np.append(imagelist1, images)


# In[25]:


x=0
for imagn in imagelist:
    cac(imagn,3)7


# In[22]:


cacimg=cv2.imread('images/all_souls_000047.jpg', 1)
scale_percent = 50 # percent of original size
width = int(cacimg.shape[1] * scale_percent / 100)
height = int(cacimg.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
cacimg = cv2.resize(cacimg, dim, interpolation = cv2.INTER_AREA)
start = time.time()
matrix=cac('all_souls_000047.jpg')
print(time.time()-start)
for i in range(0, 4):
        print("k = ", 2 * i + 1)
#         print( matrix[i])


# In[ ]:


for i in range(0, 4): 
        print("k = ", 2 * i + 1)
        print(matrix[i])
matrix=np.asarray(matrix)
# matrix.size


# In[9]:





# In[29]:





# In[31]:





# In[4]:


i=0
dic=dict()

pool = Pool()                         # Create a multiprocessing Pool
COLORAUTOMAT=pool.map(cac, imagelist)
# start = time.time()
# for images in os.listdir(path):
#     i+=1
#     #     print(image_to_rotate)
#     cacimg=cv2.imread('images/'+images, 1)
#     matrix=cac(cacimg)
#     dic.update( {image : np.asarray(matrix) } )
#     print(i,"is done",time.time()-start)


# In[13]:


x=0
for imagn in imagelist:
    img=cv2.imread('images/'+imagn, 1)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)
    #     https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 64
    ret, label, center = cv2.kmeans(Z, K,None,criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
#     print(res)
    print(x)
    x+=1
    res2 = res.reshape((img.shape))
#     cv2.imwrite( str("64transformed/"+imagn[:-4]+"_TRANS"+imagn[-4:]), res2) 


# In[ ]:





# In[105]:


def CACsorder(queryimg,imagelist):
    queryimgnpy=queryimg+'.jpg.npy'
    order=[]
    for i in imagelist:
        temp=[]
        arr1= np.load( str('CACs_done_13hours/'+queryimgnpy) )
        arr2= np.load( str('CACs_done_13hours/'+i+'.npy'   ) )
        if(arr1.shape==arr2.shape):
            val=np.linalg.norm( (arr1 - arr2)/(1+arr1+arr2) ,1)
#             print(val)
            order.append( [i,val] )
    return order


# In[107]:


order=CACsorder('all_souls_000013',imagelist1)


# In[112]:


def sec(x): 
    return x[1]
neworder=sorted(order,key=sec,reverse=True)
neworder[0:3]
files=[x[0][0:-4] for x in neworder[0:10]]
files


# In[38]:



plt.imshow( Image.open('images/'+neworder[0][0] ))
plt.show()
plt.imshow( Image.open('images/'+neworder[1][0] ))
plt.show()
plt.imshow( Image.open('images/'+neworder[2][0] ))
plt.show()
plt.imshow( Image.open('images/'+neworder[3][0] ))
plt.show()
# plt.imshow( Image.open('images/'+neworder[4][0] ))
# plt.show()
# plt.imshow( Image.open('images/'+neworder[5][0] ))
# plt.show()
# plt.imshow( Image.open('images/'+neworder[6][0] ))
# plt.show()
# plt.imshow( Image.open('images/'+neworder[7][0] ))
# plt.show()
# plt.imshow( Image.open('images/'+neworder[8][0] ))
# plt.show()
# plt.imshow( Image.open('images/'+neworder[9][0] ))
# plt.show()


# In[122]:


pwd1 = 'D:/GitHub/Semester-6/Multimedia Computing and Application/Homeworks/HW1/train/query/'
pwd2 = 'D:/GitHub/Semester-6/Multimedia Computing and Application/Homeworks/HW1/train/ground_truth/'

for filename in os.listdir(pwd1):
    print(filename[:-10])
    f = open(pwd1+filename, "r")
    fn=f.read().split(" ")[0] + str('.txt')
    fn=fn.split("_")[1:3]
    fn1=fn[0]+"_"+fn[1]+"_1_good.txt"
    fn2=fn[0]+"_"+fn[1]+"_1_junk.txt"
    fn3=fn[0]+"_"+fn[1]+"_1_ok.txt"
#     print("file 1",fn1)
#     print("file 2",fn2)
#     print("file 3",fn3)
    order=CACsorder('all_souls_000013',imagelist1)
    neworder=sorted(order,key=sec,reverse=True)
    files=[x[0][0:-4] for x in neworder[0:10]]
    files
    
    comparefiles1 = list(open(pwd2+fn1, "r"))
    comparefiles1 =[ x[0:-1] for x in  comparefiles1]
    comparefiles2 = list(open(pwd2+fn2, "r"))
    comparefiles2 =[ x[0:-1] for x in  comparefiles1]
    comparefiles3 = list(open(pwd2+fn3, "r"))
    comparefiles3 =[ x[0:-1] for x in  comparefiles3]
    
    
    finalcompare=comparefiles1+comparefiles2+comparefiles3
    print("top 10",files)
    print("images to be compared to",finalcompare)
    print(  set(files).intersection( list(finalcompare) )  )
    
    print("\n")


# In[ ]:





# In[ ]:





# # testing

# In[ ]:


from scipy import ndimage, misc
import matplotlib.pyplot as plt
ascent = misc.ascent()
plt.imshow(ascent)


# In[ ]:


fig = plt.figure()
plt.gray()  # show the filtered result in grayscale
# ax1 = fig.add_subplot(121)  # left side
# ax2 = fig.add_subplot(122)  # right side
plt.show()


# In[ ]:


result = ndimage.gaussian_laplace(ascent, sigma=1)
plt.imshow(result)


# In[104]:


plt.imshow(result)
plt.show()


# In[107]:


[i for i in [1,3,5,7]]==[1,3,5,7]


# In[118]:


unique( np.asarray([ [1,2,3,4] ,[1,2,3,4],[1,2,3,4] ]) ) 


# In[153]:


c1=cac('all_souls_000047.jpg')
c1


# In[ ]:


start=time.time()
image = cv2.imread('images/all_souls_000047.jpg')
(h, w) = image.shape[:2]
# convert the image from the RGB color space to the L*a*b*
# color space -- since we will be clustering using k-means
# which is based on the euclidean distance, we'll use the
# L*a*b* color space where the euclidean distance implies
# perceptual meaning
image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
# reshape the image into a feature vector so that k-means
# can be applied
image = image.reshape((image.shape[0] * image.shape[1], 3))
# apply k-means using the specified number of clusters and
# then create the quantized image based on the predictions
clt = MiniBatchKMeans(n_clusters = 64)
labels = clt.fit_predict(image)
quant = clt.cluster_centers_.astype("uint8")[labels]
# reshape the feature vectors to images
quant = quant.reshape((h, w, 3))
image = image.reshape((h, w, 3))
# convert from L*a*b* to RGB
quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
print(time.time()-start)
# display the images and wait for a keypress
cv2.imshow("image", np.hstack([image, quant]))
cv2.waitKey(0)


# In[8]:


tempo=np.random.rand(10,10)
tempo


# In[26]:


len( getNeighbors( tempo.shape[1],tempo.shape[0],4,4,4) )


# In[41]:


hash(tuple([1,2,3]))


# In[45]:


np.asarray([])


# In[ ]:


img = np.zeros([5,5,3])

img[:,:,0] = np.ones([5,5])*64/255.0
img[:,:,1] = np.ones([5,5])*128/255.0
img[:,:,2] = np.ones([5,5])*192/255.0

cv2.imwrite('color_img.jpg', img)
cv2.imshow("image", img);
cv2.waitKey();


# In[163]:


np.zeros([5,5,3])

[200,200,200]
img=np.asarray([ [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
             [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
             [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
             [[0,0,0],[200,200,200],[200,200,200],[0,0,0],[0,0,0],[200,200,200],[200,200,200],[0,0,0]],
             [[0,0,0],[200,200,200],[200,200,200],[0,0,0],[0,0,0],[200,200,200],[200,200,200],[0,0,0]],
             [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
             [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
             [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]])
d=[1,2,3,4,5,6]
mat,freq=correlogram(img,[[0,0,0],[200,200,200]],d)
mat=np.asarray(mat)
mat


# In[164]:


freq=[x / len(d) for x in freq]
freq


# In[192]:


final=[]
x=0
for i in mat:
    final.append( [i / (j*8*d[x]) for i, j in zip(i, freq)]  )
    x+=1
np.asarray(final)

plt.plot(d,[i[0] for i in final])
plt.plot(d,[i[0] for i in final],'*')
plt.plot(d,[i[1] for i in final])
plt.plot(d,[i[1] for i in final],'o')


# ![image.png](attachment:image.png)

# In[166]:



mat


# In[142]:


[3,3]/[2,2]


# In[116]:




