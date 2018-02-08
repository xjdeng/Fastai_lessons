from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *

class BinaryClassifier(object):
    
    def __init__(self, PATH, sz = 224, arch = resnet34):
        self.PATH = PATH
        self.sz = 224
        self.arch = arch
        