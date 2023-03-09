import os
import pydicom
import cv2
import imageio as io
import numpy as np
from pixelhop2 import Pixelhop2
from skimage.util import view_as_windows


import pathlib
train_dir = "./data/img/dicom"
#test_dir = "/u/scratch/e/ewolos/test"


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

    for file in os.listdir(train_dir):
        filename = train_dir + '/' + file
        images.append(dicom_to_png(filename))

    X = np.reshape(images, (len(images), 206, 206, 1))

    # set args
    SaabArgs = [{'num_AC_kernels': 4, 'needBias':False, 'useDC':True, 'batch':None}, 
                {'num_AC_kernels': 4, 'needBias':True, 'useDC':True, 'batch':None}]
    shrinkArgs = [{'func':Shrink, 'win':3}, 
                {'func': Shrink, 'win':2},]
    concatArg = {'func':Concat}
    p2 = Pixelhop2(depth=2, TH1=0.1, TH2=0.00005, SaabArgs=SaabArgs, shrinkArgs=shrinkArgs, concatArg=concatArg)
    output = p2.fit(X)
    output = p2.transform(X)

    features = []

    for i in range(len(output[1])):
        x = (output[1][i][0]).tolist()
        x.append([0,0,0,0])
        x.append([0,0,0,0])
        x = np.asarray(x)
        y = np.zeros(shape=(36, 2048))
        features.append((x, y, (206, 206)))

    return features

if __name__ == "__main__":
    feat_1, feat_2, (img_w, img_h) = extract_all_features()[0]
    print(feat_1.shape, feat_2.shape, (img_w, img_h))



