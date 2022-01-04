# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 19:17:03 2021

@author: Ross
"""

import numpy as np
import matplotlib.pyplot as plt

from osgeo import gdal

from numpy import nanmin, nanmax, where, isnan, fabs
from numpy import nanmean, nanstd

def stddev_stretch(b):
    bf = b.flatten()
    m = nanmean(bf)
    s = nanstd(bf)
    min_ = m - 2*s
    max_ = m + 2*s
    min_ = max(nanmin(bf), min_)
    max_ = min(nanmax(bf), max_)
    return (255.99*(b-min_)/(max_-min_)).clip(0, 255).astype('u1')




import skimage.filters 
def getThreshold(band_data, percent=0.1):
    # calculate threshold using Otsu method
    threshold_otsu = skimage.filters.threshold_otsu(band_data)
    # calculate threshold using minimum method
    threshold_minimum = skimage.filters.threshold_minimum(band_data)
    # get number of pixels for both thresholds
    numPixOtsu = len(band_data[abs(band_data - threshold_otsu) < percent])
    numPixMinimum = len(band_data[abs(band_data - threshold_minimum) < percent])

    # if number of pixels at minimum threshold is less than 0.1% of number of pixels at Otsu threshold
    if abs(numPixMinimum/numPixOtsu) < percent/100.0:
        # adjust band data according
        if threshold_otsu < threshold_minimum:
            band_data = band_data[band_data < threshold_minimum]
            threshold_minimum = skimage.filters.threshold_minimum(band_data)
        else:
            band_data = band_data[band_data > threshold_minimum]
            threshold_minimum = skimage.filters.threshold_minimum(band_data)
        numPixMinimum = len(band_data[abs(band_data - threshold_minimum) < percent])
    # check for final threshold
    if abs(numPixMinimum/numPixOtsu) < percent/100.0:
        threshold = threshold_otsu
    else:
        threshold = threshold_minimum

    return threshold



def plotBand(band_data, threshold, binary=False):
    # color stretch
    vmin, vmax = 0, 1
    # read pixel values
    w,h=band_data.shape
    # color stretch
    if binary:
        cmap = plt.get_cmap('binary')
    else:
        vmin = np.percentile(band_data, 2.5)
        vmax = np.percentile(band_data, 97.5)
        cmap = plt.get_cmap('gray')
    # plot band
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16,6))
    ax1.imshow(band_data, cmap=cmap, vmin=vmin, vmax=vmax)
    #ax1.set_title(band.getName())
    # plot histogram
    band_data.shape = h * w 
    ax2.hist(np.asarray(band_data[band_data != 0], dtype='float'), bins=2048)
    ax2.axvline(x=threshold, color='r')
    #ax2.set_title('Histogram: %s' % band.getName())
    
    for ax in fig.get_axes():
        ax.label_outer()


def getExtremePoints(data, typeOfInflexion = None, maxPoints = None):
    """
    This method returns the indeces where there is a change in the trend of the input series.
    typeOfInflexion = None returns all inflexion points, max only maximum values and min
    only min,
    """
    a = np.diff(data)
    asign = np.sign(a)
    signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
    idx = np.where(signchange ==1)[0]

    if typeOfInflexion == 'max' and data[idx[0]] < data[idx[1]]:
        idx = idx[1:][::2]
        
    elif typeOfInflexion == 'min' and data[idx[0]] > data[idx[1]]:
        idx = idx[1:][::2]
    elif typeOfInflexion is not None:
        idx = idx[::2]
    
    # sort ids by min value
    if 0 in idx:
        idx = np.delete(idx, 0)
    if (len(data)-1) in idx:
        idx = np.delete(idx, len(data)-1)
    idx = idx[np.argsort(data[idx])]
    # If we have maxpoints we want to make sure the timeseries has a cutpoint
    # in each segment, not all on a small interval
    if maxPoints is not None:
        idx= idx[:maxPoints]
        if len(idx) < maxPoints:
            return (np.arange(maxPoints) + 1) * (len(data)//(maxPoints + 1))
    
    return idx



from tqdm import tqdm



from scipy.ndimage.filters import generic_filter
import numpy as np
from scipy.stats import mode

def filter_function(invalues):
   invalues_mode = mode(invalues, axis=None, nan_policy='omit')
   return invalues_mode[0]

function = lambda array: generic_filter(array, function=filter_function, size=3)



# Read in our image and ROI image
img_ds = gdal.Open(r'E:\Ahmedabad_Uni(SEAS)\ISRO_Internship\Project1_Mini\N24E070_17_MOS_F02DAR\N24E070_17_sl_HH_F02DAR', gdal.GA_ReadOnly) 
img = img_ds.GetRasterBand(1).ReadAsArray()
#img[np.isnan(img)]=0.0
#img_fnf = img_ds_fnf.GetRasterBand(1).ReadAsArray()

img_tile=[]
k=2250;
for i in range(img.shape[0]//k):
    for j in range(img.shape[1]//k):
        img_tile.append(img[i*k:(i+1)*k,j*k:(j+1)*k])

cmap = plt.get_cmap('gray')
import matplotlib.pyplot as plt
plt.figure()
for i in range(len(img_tile)):
    plt.subplot(2,2,i+1)
    plt.imshow(stddev_stretch(img_tile[i]), cmap)
    
'''
######### Display Image ##################
vmin = np.percentile(img, 2.5)
vmax = np.percentile(img, 97.5)
cmap = plt.get_cmap('gray')

plt.figure()
plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
plt.title('Image')
'''


################################################################################
# # If image is in linear scale do the following linear = 10.0**(img/10.0);
#img = 10*np.log10(img);
# cf=83

##################################################################################


preprocessed=1
if preprocessed:
    # Calibration operator
    print('1. Radiometric Calibration:   ', end='')
    img=img.astype(np.float32)
    img_cal_db = 20*np.log10(img)-83;
    
    
    # Speckle-Filter operator
    print('2. Speckle Filtering:         ', end='', flush=True)
    from scipy.ndimage.filters import median_filter
    def medianFilter(array, size=3):
        return median_filter(array,size=size)
        
    img_cal_db_sp=medianFilter(img_cal_db,size=5)
    img_cal_db=img_cal_db_sp.copy()

else:
    img_cal_db=img.copy()



######################################################
########## The following code finds the number of modes ######

lcNew = img_cal_db.flatten('C')
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,5))
y, x = np.histogram(lcNew, bins=512,range=[-30,-1], density=1)
ax1.plot(x[0:-1], y)  # <- or here
# smooth
from scipy.ndimage import gaussian_filter1d
smooth = gaussian_filter1d(y, 3)
ax2.plot(x[0:-1], smooth)  # <- or here



#Finding inflection points a given signal
#infls = np.diff(np.sign(np.diff(smooth))).nonzero()[0] + 1 # local min+max
infls_min = (np.diff(np.sign(np.diff(smooth))) > 0).nonzero()[0] + 1 # local min
infls = (np.diff(np.sign(np.diff(smooth))) < 0).nonzero()[0] + 1 # local max
print("The number of modes is found to be :",len(infls))

plt.figure()
# # compute second derivative
smooth_d1 = np.diff(smooth)
plt.plot(smooth/np.max(smooth), label='Smoothed Data')
#plt.plot(np.gradient(true_pdf)/np.max(np.gradient(true_pdf)), label='First Derivative')
plt.plot(smooth_d1/np.max(smooth_d1), label='First Derivative (scaled)')
for i, infl in enumerate(infls, 1):
    plt.axvline(x=infl, color='k', label=f'Inflection Point {i}')
plt.axhline(y = 0.0, color = 'r', linestyle = '-')


print('Thresholding:         ', end='', flush=True)
threshold = getThreshold(img_cal_db,percent=0.1)
print(threshold)


binary=np.zeros(img_cal_db.shape)
binary[img_cal_db>threshold]=1

# color stretch
vmin = np.percentile(img_cal_db, 2.5)
vmax = np.percentile(img_cal_db, 97.5)
cmap = plt.get_cmap('gray')

plt.figure(figsize=(20,20))
ax1=plt.subplot(121)
ax1.imshow(img_cal_db, cmap=cmap, vmin=vmin, vmax=vmax)
#plt.imshow(stddev_stretch(img), cmap='gray')
plt.title('ALOS Data')
ax2=plt.subplot(122,sharex=ax1,sharey=ax1)
ax2.imshow(binary, cmap=plt.get_cmap('Blues_r'), vmin=0, vmax=1)
plt.title('Water')
plt.show()



plt.figure(figsize=(10,10))
band_data=img_cal_db.flatten()
plt.hist(np.asarray(band_data[band_data != 0], dtype='float'), bins=512)
plt.axvline(x=threshold, color='r')
plt.title('Threshold')
plt.show()



# from sklearn.mixture import GaussianMixture
# n_components = np.arange(1, 10)
# clfs = [GaussianMixture(n, max_iter = 1000).fit(img) for n in n_components]
# bics = [clf.bic(img) for clf in clfs]
# aics = [clf.aic(img) for clf in clfs]

# plt.plot(n_components, bics, label = 'BIC')
# plt.plot(n_components, aics, label = 'AIC')
# plt.xlabel('n_components')
# plt.legend()
# plt.show()



