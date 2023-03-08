# 2020.04.14
import numpy as np 
import importlib.util
import sys
spec = importlib.util.spec_from_file_location("cwSaab", "/content/drive/MyDrive/PixelHop2/cwSaab.py")
cwSaab = importlib.util.module_from_spec(spec)
sys.modules["cwSaab"] = cwSaab
spec.loader.exec_module(cwSaab)
from cwSaab import cwSaab

from sys import argv

class Pixelhop2(cwSaab):
    def __init__(self, depth=1, TH1=0.01, TH2=0.001, SaabArgs=None, shrinkArgs=None, concatArg=None, splitMode=2):
        super().__init__(depth=depth, energyTH=TH1, SaabArgs=SaabArgs, shrinkArgs=shrinkArgs, concatArg={'func':lambda X, concatArg: X}, splitMode=splitMode)
        self.TH1 = TH1
        self.TH2 = TH2
        self.concatArg = concatArg
        #self.kernelRetain=kernelRetain

    def select_(self, X):#(depth,N,S,S,K)
        for i in range(self.depth):
            X[i] = X[i][:, :, :, self.Energy[i] >= self.TH2]
        
        return X

    def fit(self, X):
        X = super().fit(X)
        return self

    def transform(self, X):
        X = super().transform(X)
        X = self.select_(X)
        return self.concatArg['func'](X, self.concatArg)
