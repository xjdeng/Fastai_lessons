from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *

from path import Path as path
import warnings

class BinaryClassifier(object):
    
    def __init__(self, PATH, sz = 224, arch = resnet34):
        self.PATH = path(PATH)
        self.sz = 224
        self.arch = arch
        
        self._check_inputs()
    
    
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
        self.class1 = str(traindirs[0])
        self.class2 = str(traindirs[1])
        if (path(self.valid + self.class1).exists() == False) or \
        (path(self.valid + self.class2).exists() == False):
            raise(IOError("No matching validation class directories found"))
        
        