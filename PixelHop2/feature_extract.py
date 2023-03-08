import os
import pydicom
import cv2
import imageio as io
import numpy as np
from pixelhop2 import Pixelhop2
from skimage.util import view_as_windows

train_dir = "/u/scratch/e/ewolos/train"
test_dir = "/u/scratch/e/ewolos/test"


def dicom_to_png(path):
    ds = pydicom.dcmread(path)
    im = ds.pixel_array
    im = cv2.resize(im, (206, 206))
    return im

# example callback function for collecting patches and its inverse
def Shrink(X, shrinkArg):
    win = shrinkArg['win']
    X = view_as_windows(X, (1,win,win,1), (1,win,win,1))
    return X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)

# example callback function for how to concate features from different hops
def Concat(X, concatArg):
    return X


def extract_all_features():

    images = []

    for file in os.scandir(train_dir):
        filename = train_dir + file.name
        images.append(dicom_to_png(filename))

    X = np.reshape(images, (len(images), 206, 206, 1))

    # set args
    SaabArgs = [{'num_AC_kernels': 8, 'needBias':False, 'useDC':True, 'batch':None}, 
                {'num_AC_kernels': 8, 'needBias':True, 'useDC':True, 'batch':None}]
    shrinkArgs = [{'func':Shrink, 'win':3}, 
                {'func': Shrink, 'win':3},]
    concatArg = {'func':Concat}
    p2 = Pixelhop2(depth=2, TH1=0.1, TH2=0.00005, SaabArgs=SaabArgs, shrinkArgs=shrinkArgs, concatArg=concatArg)
    output = p2.fit(X)
    output = p2.transform(X)
    return (output[0], output[1])



