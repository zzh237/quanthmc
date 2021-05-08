from abc import ABC, abstractclassmethod
import numpy as np 
import torch.nn as nn 

class exp_interface(ABC):
    def prepare_model(self)->nn.Module:
        raise NotImplementedError
    def generate_data(self)->dict:
        raise NotImplementedError 
    def get_algorithm(self):
        raise NotImplementedError