import os 
import numpy as np 
from experiments import exp_interface
from data import * 
from model.fully_connected import * 
from model.vqc_layers import * 
from algo.hmc import * 
from quant_architecture.variational_qunt_circuit import * 

from algo.algo_interface import *


class mnist_exp(exp_interface):
    def __init__(self):
        self.filename = os.path.basename(__file__) 

    def prepare_exp(self, args, data):

        self.args = args 
        self.data = data  

    # this we do rather use it here
    def prepare_model(self)->nn.Module:     
        if self.args.model_name == 'Quant':   
            model = vqc_net(self.args, vqc(self.args))
        if self.args.model_name == 'mlp':
            model = MLP(input_dim=784, width=self.args.mlp_width, depth=self.args.mlp_depth, output_dim=10)
        return model 
        
    # @getattr
    # def model_name(self):
    #     return repr(self.algo)
   
    # we don't need this line of codes, since they are all the same for each algorithm
    def apply_algorithm(self, algo_interface):
        self.algo = algo_interface
        
        if self.args.algo_name == 'HMC':
            self.algo.fit()
        acc, loss = self.algo.predict()
        
        return acc, loss 
    
    
        
    