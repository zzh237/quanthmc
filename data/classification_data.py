from abc import ABC, abstractclassmethod
import numpy as np 
import torch.nn as nn 

class classification_data(ABC):
    def load_data(self):
	    raise NotImplementedError

    def train_test_data(self):
        raise NotImplementedError

    
    
