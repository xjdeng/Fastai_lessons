from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *

import numpy as np
from path import Path as path
import warnings

class BinaryClassifier(object):
    
    def __init__(self, PATH, sz = 224, arch = resnet34):
        self.PATH = path(PATH)
        self.sz = 224
        self.arch = arch
        
        self._check_inputs()
        self._build_classifier()
    
    def _build_classifier(self):
        self.data = ImageClassifierData.from_paths(self.PATH, tfms=\
                                                   tfms_from_model(self.arch,\
                                                                  self.sz))
        self.learn = ConvLearner.pretrained(self.arch, self.data, \
                                            precompute=True)

    
    def _check_inputs(self):
        self.train = path(self.PATH + "/train/")
        if self.train.exists() == False:
            raise(IOError("No training directory found."))
        self.valid = path(self.PATH + "/valid/")
        if self.valid.exists() == False:
            raise(IOError("No validation directory found."))
        traindirs = self.train.dirs()
        if len(traindirs) > 2:
            warnings.warn("More than 2 directories found in training dir.")
        elif len(traindirs) < 2:
            raise(IOError("Only 1 class directory found in training dir."))
        self.class1 = str(traindirs[0].name)
        self.class2 = str(traindirs[1].name)
        if (path(self.valid + self.class1).exists() == False) or \
        (path(self.valid + self.class2).exists() == False):
            raise(IOError("No matching validation class directories found"))
        

    def fit(self, lr = 0.01, epochs = 2):
        self.learn.fit(lr, epochs)
        
    def predict(self):
        log_preds = self.learn.predict()
        self.preds = np.argmax(log_preds, axis=1)
        self.probs = np.exp(log_preds[:,1])