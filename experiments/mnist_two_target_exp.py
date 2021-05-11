import os 
import numpy as np 
from experiments import exp_interface
from data import * 
from model.fully_connected import * 
from model.apc_layers import * 
from algo.hmc import * 
from quant_architecture.amplitude_circuit import * 

from algo.algo_interface import *


class mnist_two_target_exp(exp_interface):
    def __init__(self):
        self.filename = os.path.basename(__file__) 

    def prepare_exp(self, args):

        self.args = args 
       

    # this we do rather use it here
    def prepare_model(self)->nn.Module:  
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        if self.args.model_name == 'Quant':   
            model = apc_net(self.args, ampc(self.args))
        if self.args.model_name == 'mlp':
            model = MLP(input_dim=1024, width=self.args.mlp_width, depth=self.args.mlp_depth, output_dim=1).to(device)
        return model 
        
    def feed_data(self, data):
        self.data = data  
    
    
    # @getattr
    # def model_name(self):
    #     return repr(self.algo)
   
    # we don't need this line of codes, since they are all the same for each algorithm
    def apply_algorithm(self, algo_interface):
        self.algo = algo_interface
        
        if self.args.algo_name == 'HMC':
            self.algo.fit()
        err, best_err,loss = self.algo.predict()
        
        return err, best_err,loss  
    
    
        
    
