#!/usr/bin/env python
# coding: utf-8

# In[20]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math


# In[21]:


def convolution(img, kernel):
    #flip kernel for convolution
    kernel_r = np.flipud(np.fliplr(kernel))
    #find the amount of padding needed for each side of the image
    padx = kernel.shape[0]
    pady = kernel.shape[1]
    
    #making frame for new padded image
    image_pad = np.zeros((img.shape[0] + padx - 1,img.shape[1]+pady - 1), dtype = np.float32) 
    
    #leave the padded area and copy over the other information.
    image_pad[round((padx-1)/2):round(-(padx-1)/2), round((pady-1)/2):round(-(pady-1)/2)] = img[0:,0:] 
    
    #make an empty frame for the output
    img_out = np.zeros_like(img, dtype = np.float32)
    
    #if image is 3-D do three layers of convolution on X, Y, Z
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            #for z in range(img.shape[2]): 
                #make sure the sum value is within range of 0 - 255
                img_out[x,y] = (image_pad[x:x+padx, y:y+pady]*kernel_r).sum()
    img_out = img_out / img_out.max() *255

    return img_out


# In[22]:


def gaussian(sig, x, y):
    return ( (1./(2.*np.pi*(sig**2)))*np.exp(-(x**2 + y**2)/(2.*sig**2)) )

def Gaussian_kernel(sig, x_dim, y_dim):
    
    gaussian_kernel = np.zeros(shape=(x_dim, y_dim),dtype = np.float32)
    x_dim = x_dim // 2
    y_dim = y_dim // 2
    for x in range(-x_dim, x_dim+1):
        for y in range(-y_dim, y_dim+1):
            gaussian_kernel[x+x_dim][y+ y_dim] = gaussian(sig,x,y)
    return gaussian_kernel/gaussian_kernel.sum()


def GaussianSmoothing(i, k_size, sigma):
    kernel = Gaussian_kernel(sigma, k_size, k_size)
    return convolution(i,kernel)


# In[23]:



def ImageGradient(img):
#defind x and y kernels to calculate the vertical and horizontal differences and derivation
    x_kernel = np.array([
        [-1,0,1],
        [-2,0,2],
        [-1,0,1]
    ], np.float32)
    y_kernel = np.array([
        [-1,-2,-1],
        [ 0, 0, 0],
        [ 1, 2, 1]
    ], np.float32)
    ix = convolution(img, x_kernel)
    iy = convolution(img, y_kernel)
    
    
    mag = np.zeros(img.shape, np.float32)
    theta = np.zeros(img.shape, np.float32)
    #Calculate
    for x in range(img.shape[0]):
        for y in range(img.shape[1]): 
                #make sure the sum value is within range of 0 - 255
                mag[x,y] = math.sqrt(int(ix[x,y]**2) + int(iy[x,y]**2))
                
    theta = np.arctan2(iy,ix)
    mag = mag/mag.max() * 255
    return mag, theta


# In[24]:


def NonmaximaSuppress(Mag, Theta):
    Theta = (Theta*180)/np.pi

    nonMax = np.zeros(Mag.shape, dtype = np.int32)

    
    for x in range(1, Mag.shape[0]-1):
        for y in range(1,Mag.shape[1]-1):
                p = 255 # left gradient 
                r = 255 # right gradient
                
                if Theta[x,y] < 0:
                    Theta[x,y] += 180
                t = Theta[x,y]
                    
                #at 0 ^ 360 degree ( less than half of 45 and more than 135+22.5)
                if ( t >= 0 and t < 22.5 ) or (t >= 157.5 and t <= 180):
                    p = Mag[x,y+1]
                    r = Mag[x,y-1]
                    
                #at 45 ^ 325 degree
                elif( t >= 22.5 and t < 67.5):
                    p = Mag[x+1,y-1]
                    r = Mag[x-1,y+1]
                    
                #at 90 ^ 270 degree
                elif( t >= 67.5 and t < 112.5):
                    p = Mag[x+1,y]
                    r = Mag[x-1,y]
                    
                #at 135 ^ 225
                elif( t >= 112.5 and t < 157.5):
                    p = Mag[x-1,y-1]
                    r = Mag[x+1,y+1]
                    
                if p <= Mag[x,y] and r <= Mag[x,y]:
                    nonMax[x,y] = Mag[x,y]
                else :
                    nonMax[x,y] = 0
    return nonMax
                    


# In[25]:


def EdgeLinking(Mag, Mag_weak, Mag_strong):
    High = Mag.max() * Mag_strong
    Low = Mag.max() * Mag_weak
    result = np.zeros(Mag.shape, dtype =np.int32)
    H=255
    L=100
    for x in range( Mag.shape[0]):
        for y in range(Mag.shape[1]):
            if Mag[x,y] >= High:
                result[x,y] = H
            elif Mag[x,y] >= Low :
                result[x,y] = L
            else:
                result[x,y] = 0
                
    for x in range( 1,Mag.shape[0]-1):
        for y in range(1,Mag.shape[1]-1):
            if result[x,y] == L:
                if ((result[x-1,y-1] ==H) or (result[x-1,y] ==H) or (result[x-1,y+1] ==H )or 
                    (result[x,y-1] ==H) or (result[x,y+1] ==H) or (result[x+1,y-1] ==H) or 
                    (result[x+1,y] ==H) or (result[x+1,y+1] ==H)):
                        result[x,y] = H
                else:
                    result[x,y] = 0
                        
    return result
    


# In[26]:


img_in = cv2.imread('lena_gray.png',0)
print("Loading lena_gray.png")

# plt.imshow(img_in, cmap = "gray")
# plt.show()


# In[27]:


print("gaussian size 3 sigma 1")
img1 = GaussianSmoothing(img_in, 3, 1)

Mag11, Theta1 = ImageGradient(img1)

Mag12 =  NonmaximaSuppress(Mag11, Theta1)

Mag13 = EdgeLinking(Mag12, 0.05, .1)


# plt.imshow(img1,cmap = 'gray')
# plt.show()

# plt.imshow(Mag11, cmap = "gray")
# plt.show()

# plt.imshow(Theta1, cmap = "gray")
# plt.show()

# plt.imshow(Mag12, cmap = "gray")
# plt.show()

# plt.imshow(Mag13, cmap = "gray")
# plt.show()


# In[28]:


print("gaussian size 5 sigma 5")
img2 = GaussianSmoothing(img_in, 5, 5)

Mag21, Theta2 = ImageGradient(img2)

Mag22 =  NonmaximaSuppress(Mag21, Theta2)
# Mag2 = 4*Mag2 / Mag2.max() *255

Mag23 = EdgeLinking(Mag22, 0.05, .1)


# plt.imshow(img2,cmap = 'gray')
# plt.show()

# plt.imshow(Mag21, cmap = "gray")
# plt.show()

# plt.imshow(Theta2, cmap = "gray")
# plt.show()

# plt.imshow(Mag22, cmap = "gray")
# plt.show()

# plt.imshow(Mag23, cmap = "gray")
# plt.show()


# In[29]:


print("gaussian size 5 sigma 5 with different Edge linking tresholds ")
img3 = GaussianSmoothing(img_in, 5, 5)

Mag31, Theta3 = ImageGradient(img3)

Mag32 =  NonmaximaSuppress(Mag31, Theta3)
# Mag3 = 4*Mag3 / Mag3.max() *255

Mag33 = EdgeLinking(Mag32, 0.4, .8)


# plt.imshow(img3,cmap = 'gray')
# plt.show()

# plt.imshow(Mag31, cmap = "gray")
# plt.show()

# plt.imshow(Theta3, cmap = "gray")
# plt.show()

# plt.imshow(Mag32, cmap = "gray")
# plt.show()

# plt.imshow(Mag33, cmap = "gray")
# plt.show()


# In[30]:


print("gaussian size 11 sigma 10 with lower tresholds")
img4 = GaussianSmoothing(img_in, 11, 10)

Mag41, Theta4 = ImageGradient(img4)

Mag42 =  NonmaximaSuppress(Mag41, Theta4)
# Mag4 = 4*Mag4 / Mag4.max() *255

Mag43 = EdgeLinking(Mag42, 0.05, .1)


# plt.imshow(img4,cmap = 'gray')
# plt.show()

# plt.imshow(Mag41, cmap = "gray")
# plt.show()

# plt.imshow(Theta4, cmap = "gray")
# plt.show()

# plt.imshow(Mag42, cmap = "gray")
# plt.show()

# plt.imshow(Mag43, cmap = "gray")
# plt.show()


# In[31]:


cv2.imwrite("Gauss-lena-gaus3Sig1-lowTresh.png",img1)
cv2.imwrite("Gauss-lena-gaus5Sig5-lowTresh.png",img2)
cv2.imwrite("Gauss-lena-gaus5Sig5-highTresh.png",img3)
cv2.imwrite("Gauss-lena-gaus11Sig10-lowTresh.png",img4)

cv2.imwrite("Grad-lena-gaus3Sig1-lowTresh.png",Mag11)
cv2.imwrite("Grad-lena-gaus5Sig5-lowTresh.png",Mag21)
cv2.imwrite("Grad-lena-gaus5Sig5-highTresh.png",Mag31)
cv2.imwrite("Grad-lena-gaus11Sig10-lowTresh.png",Mag41)

cv2.imwrite("MaxSupp-lena-gaus3Sig1-lowTresh.png",Mag12)
cv2.imwrite("MaxSupp-lena-gaus5Sig5-lowTresh.png",Mag22)
cv2.imwrite("MaxSupp-lena-gaus5Sig5-highTresh.png",Mag32)
cv2.imwrite("MaxSupp-lena-gaus11Sig10-lowTresh.png",Mag42)

cv2.imwrite("Result-lena-gaus3Sig1-lowTresh.png",Mag13)
cv2.imwrite("Result-lena-gaus5Sig5-lowTresh.png",Mag23)
cv2.imwrite("Result-lena-gaus5Sig5-highTresh.png",Mag33)
cv2.imwrite("Result-lena-gaus11Sig10-lowTresh.png",Mag43)


# In[32]:


img_in2 = cv2.imread('test.png',0)
print("Loading Test.png")
# plt.imshow(img_in2, cmap = "gray")
# plt.show()


# In[33]:


print("gaussian size 3 sigma 0.5")
img1 = GaussianSmoothing(img_in2, 3, 0.5)

Mag11, Theta1 = ImageGradient(img1)

Mag12 =  NonmaximaSuppress(Mag11, Theta1)
# Mag1 = 4*Mag1 / Mag1.max() *255

Mag13 = EdgeLinking(Mag12, 0.05, .1)


# plt.imshow(img1,cmap = 'gray')
# plt.show()

# plt.imshow(Mag11, cmap = "gray")
# plt.show()

# plt.imshow(Theta1, cmap = "gray")
# plt.show()

# plt.imshow(Mag12, cmap = "gray")
# plt.show()

# plt.imshow(Mag13, cmap = "gray")
# plt.show()


# In[34]:


print("gaussian size 5 sigma 5")
img2 = GaussianSmoothing(img_in2, 5, 5)

Mag21, Theta2 = ImageGradient(img2)

Mag22 =  NonmaximaSuppress(Mag21, Theta2)
# Mag2 = 4*Mag2 / Mag2.max() *255

Mag23 = EdgeLinking(Mag22, 0.05, .1)


# plt.imshow(img2,cmap = 'gray')
# plt.show()

# plt.imshow(Mag21, cmap = "gray")
# plt.show()

# plt.imshow(Theta2, cmap = "gray")
# plt.show()

# plt.imshow(Mag22, cmap = "gray")
# plt.show()

# plt.imshow(Mag23, cmap = "gray")
# plt.show()


# In[35]:


print("gaussian size 5 sigma 5 with different Edge linking")
img3 = GaussianSmoothing(img_in2, 5, 5)

Mag31, Theta3 = ImageGradient(img3)

Mag32 =  NonmaximaSuppress(Mag31, Theta3)
# Mag3 = 4*Mag3 / Mag3.max() *255

Mag33 = EdgeLinking(Mag32, 0.4, .8)


# plt.imshow(img3,cmap = 'gray')
# plt.show()

# plt.imshow(Mag31, cmap = "gray")
# plt.show()

# plt.imshow(Theta3, cmap = "gray")
# plt.show()

# plt.imshow(Mag32, cmap = "gray")
# plt.show()

# plt.imshow(Mag33, cmap = "gray")
# plt.show()


# In[36]:


print("gaussian size 11 sigma 10 with lower tresholds")
img4 = GaussianSmoothing(img_in2, 11, 10)

Mag41, Theta4 = ImageGradient(img4)

Mag42 =  NonmaximaSuppress(Mag41, Theta4)
# Mag4 = 4*Mag4 / Mag4.max() *255

Mag43 = EdgeLinking(Mag42, 0.05, .1)


# plt.imshow(img4,cmap = 'gray')
# plt.show()

# plt.imshow(Mag41, cmap = "gray")
# plt.show()

# plt.imshow(Theta4, cmap = "gray")
# plt.show()

# plt.imshow(Mag42, cmap = "gray")
# plt.show()

# plt.imshow(Mag43, cmap = "gray")
# plt.show()


# In[37]:


cv2.imwrite("Gauss-test-gaus3Sig1-lowTresh.png",img1)
cv2.imwrite("Gauss-test-gaus5Sig5-lowTresh.png",img2)
cv2.imwrite("Gauss-test-gaus5Sig5-highTresh.png",img3)
cv2.imwrite("Gauss-test-gaus11Sig10-lowTresh.png",img4)

cv2.imwrite("Grad-test-gaus3Sig1-lowTresh.png",Mag11)
cv2.imwrite("Grad-test-gaus5Sig5-lowTresh.png",Mag21)
cv2.imwrite("Grad-test-gaus5Sig5-highTresh.png",Mag31)
cv2.imwrite("Grad-test-gaus11Sig10-lowTresh.png",Mag41)

cv2.imwrite("MaxSupp-test-gaus3Sig1-lowTresh.png",Mag12)
cv2.imwrite("MaxSupp-test-gaus5Sig5-lowTresh.png",Mag22)
cv2.imwrite("MaxSupp-test-gaus5Sig5-highTresh.png",Mag32)
cv2.imwrite("MaxSupp-test-gaus11Sig10-lowTresh.png",Mag42)

cv2.imwrite("Result-test-gaus3Sig1-lowTresh.png",Mag13)
cv2.imwrite("Result-test-gaus5Sig5-lowTresh.png",Mag23)
cv2.imwrite("Result-test-gaus5Sig5-highTresh.png",Mag33)
cv2.imwrite("Result-test-gaus11Sig10-lowTresh.png",Mag43)

