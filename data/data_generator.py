import numpy as np 
import torch 

class data_generator():
    def __init__(self, classification_data):
        self.data =  classification_data.train_test_data()
    
    
