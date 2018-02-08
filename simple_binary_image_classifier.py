from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *

import numpy as np
#import matplotlib.pyplot as plt
from path import Path as path
import warnings

class BinaryClassifier(object):
    
    def __init__(self, PATH, sz = 224, arch = resnet34, tfms = None):
        self.PATH = path(PATH)
        self.sz = 224
        self.arch = arch
        if tfms is None:
            self.tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on,\
                                        max_zoom=1.1)
        else:
            self.tfms = tfms
        
        self._check_inputs()
        self._build_classifier()
    
    def _build_classifier(self):
        self.data = ImageClassifierData.from_paths(self.PATH, tfms=\
                                                   self.tfms)
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

    def _load_img_id(self, ds, idx):
        return np.array(PIL.Image.open(self.PATH+ds.fnames[idx]))

    def _most_by_correct(self, y, is_correct): 
        mult = -1 if (y==1)==is_correct else 1
        return self._most_by_mask(((self.preds == self.data.val_y)==is_correct)\
                                  & (self.data.val_y == y), mult)

    def _most_by_mask(self, mask, mult):
        idxs = np.where(mask)[0]
        return idxs[np.argsort(mult * self.probs[idxs])[:4]]

    def _rand_by_correct(self, is_correct):
        return rand_by_mask((self.preds == self.data.val_y)==is_correct)
    

    def fit(self, lr = 0.01, epochs = 2, **kwargs):
        self.learn.fit(lr, epochs, **kwargs)
        
    def learning_rate_finder(self):
        self.lrf = self.learn.lr_find()
        
    def plot_learning_rate(self):
        self.learn.sched.plot_lr()
        
    def plot_learning_rate_schedule(self):
        self.learn.sched.plot()
        
    def plot_most_correct1(self):
        self.plot_val_with_title(self._most_by_correct(0, True),\
                                 "Most correct " + self.class1)
        
    def plot_most_correct2(self):
        self.plot_val_with_title(self._most_by_correct(1, True),\
                                 "Most correct " + self.class2)
    def plot_most_incorrect1(self):
        self.plot_val_with_title(self._most_by_correct(0, False),\
                                 "Most incorrect " + self.class1)

    def plot_most_incorrect2(self):
        self.plot_val_with_title(self._most_by_correct(1, False),\
                                 "Most incorrect " + self.class2)

    def plot_most_uncertain(self):
        most_uncertain = np.argsort(np.abs(self.probs -0.5))[:4]
        self.plot_val_with_title(most_uncertain, \
                                 "Most uncertain predictions")

    
    def plot_random_correct(self):
        self.plot_val_with_title(self._rand_by_correct(True),\
                                 "Correctly classified")
    
    def plot_random_incorrect(self):
        self.plot_val_with_title(self._rand_by_correct(False),\
                                 "Incorrectly classified")
        
    
    
    def plot_val_with_title(self, idxs, title):
        imgs = [self._load_img_id(self.data.val_ds,x) for x in idxs]
        title_probs = [self.probs[x] for x in idxs]
        print(title)
        return self.plots(imgs, rows=1, titles=title_probs, figsize=(16,8))
    
    def plots(self, ims, figsize=(12,6), rows=1, titles=None):
        f = plt.figure(figsize=figsize)
        for i in range(len(ims)):
            sp = f.add_subplot(rows, len(ims)//rows, i+1)
            sp.axis('Off')
            if titles is not None: sp.set_title(titles[i], fontsize=16)
            plt.imshow(ims[i])        
        
    def predict(self):
        log_preds = self.learn.predict()
        self.preds = np.argmax(log_preds, axis=1)
        self.probs = np.exp(log_preds[:,1])
    
    def set_precompute(self, precompute):
        self.learn.precompute = False
        

def rand_by_mask(mask):
    return np.random.choice(np.where(mask)[0], 4, replace=False)