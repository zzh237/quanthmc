from abc import ABC, abstractclassmethod
import numpy as np 
import torch.nn as nn 

class quant_arc_interface(ABC):
    def H_layer(self, nqubits):
	# """Layer of single-qubit Hadamard gates.
    # “”“
        raise NotImplementedError


    def RY_layer(self, w):
        # """
        # Layer of parametrized qubit rotations around the y axis.
        # """
        raise NotImplementedError


    def entangling_layer(self,nqubits):
        # """
        # Layer of CNOTs followed by another shifted layer of CNOT.
        # """
        raise NotImplementedError

    def quantum_net(self, q_input_features, q_weights_flat)->list:
        # The variational quantum circuit.
        
        raise NotImplementedError
        