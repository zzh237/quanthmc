from abc import ABC, abstractclassmethod
import numpy as np 
import torch.nn as nn 

class algo_interface(ABC):
    def pred(self):
        raise NotImplementedError
    