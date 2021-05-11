from abc import ABC, abstractclassmethod
import numpy as np 
import torch.nn as nn 
from experiments.ind_exp import args



def assign_device():
    device_size = 1
    if args.data_name == 'cancer':
        device_size = 5
    if args.data_name == 'wine':
        device_size = 4
    if args.data_name == 'iris':
        device_size = 4
    if args.data_name == 'mnist':
        device_size = 10
    if args.data_name == 'mnist_two_target':
        device_size = 10
    if args.data_name == 'digits':
        device_size = 10
    return device_size


def assign_depth():
    depth_size = 1
    if args.data_name == 'cancer':
        depth_size = 1
    if args.data_name == 'wine':
        depth_size = 4
    if args.data_name == 'iris':
        depth_size = 4
    if args.data_name == 'mnist':
        depth_size = 6
    if args.data_name == 'digits':
        depth_size = 10
    return depth_size



def assign_depth():
    depth_size = 1
    if args.data_name == 'cancer':
        depth_size = 1
    if args.data_name == 'wine':
        depth_size = 4
    if args.data_name == 'iris':
        depth_size = 2
    if args.data_name == 'mnist':
        depth_size = 6
    if args.data_name == 'digits':
        depth_size = 10
    return depth_size


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
        