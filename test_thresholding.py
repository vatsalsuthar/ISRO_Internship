import cv2
import numpy as np
import argparse
import ntpath
import os
from PIL import Image
import os, os.path
from PIL import Image
#imgs = []
path_1 = 'static/Figure_1.png'
'''
valid_images = [".jpg",".gif",".png",".tga"]
for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    imgs.append(Image.open(os.path.join(path,f)))
'''
def adaptative_thresholding(path, threshold):
    
    # Load image
    I = cv2.imread(path)
    print("image is loaded")
    # Convert image to grayscale
    gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    
    # Original image size
    orignrows, origncols = gray.shape
    
    # Windows size
    M = int(np.floor(orignrows/16) + 1)
    N = int(np.floor(origncols/16) + 1)
    
    # Image border padding related to windows size
    Mextend = round(M/2)-1
    Nextend = round(N/2)-1
    
    # Padding image
    aux =cv2.copyMakeBorder(gray, top=Mextend, bottom=Mextend, left=Nextend,
                          right=Nextend, borderType=cv2.BORDER_REFLECT)
    
    windows = np.zeros((M,N),np.int32)
    
    # Image integral calculation
    imageIntegral = cv2.integral(aux, windows,-1)
    
    # Integral image size
    nrows, ncols = imageIntegral.shape
    
    # Memory allocation for cumulative region image
    result = np.zeros((orignrows, origncols))
    
    # Image cumulative pixels in windows size calculation
    for i in range(nrows-M):
        for j in range(ncols-N):
        
            result[i, j] = imageIntegral[i+M, j+N] - imageIntegral[i, j+N]+ imageIntegral[i, j] - imageIntegral[i+M,j]
     
    # Output binary image memory allocation    
    binar = np.ones((orignrows, origncols), dtype=np.bool)
    
    # Gray image weighted by windows size
    graymult = (gray).astype('float64')*M*N
    
    # Output image binarization
    binar[graymult <= result*(100.0 - threshold)/100.0] = False
    
    # binary image to UINT8 conversion
    binar = (255*binar).astype(np.uint8)
    binimg = Image.fromarray(binar)
    binimg.save('static/out_4.png', 'PNG')
    return binar, binimg


        
threshold_out = adaptative_thresholding(path_1, 0.8)
print(threshold_out)