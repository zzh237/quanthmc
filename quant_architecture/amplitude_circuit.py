import pennylane as qml

from experiments.ind_exp import args
import torch
import pennylane as qml

# from keras.datasets import mnist
import numpy as np


class ampc():

    dev = qml.device("default.qubit", wires=10)
    
    
    def __init__(self, args):
        self.args = args 
        self.args.n_qubits = 10                # Number of qubits
        self.args.q_depth = 4                 # Depth of the quantum circuit (number of variational layers)
        # self.args.q_delta = 0.01              # Initial spread of random quantum weights
        
    def H_layer(self, nqubits):
        """Layer of single-qubit Hadamard gates.
        """
        for idx in range(nqubits):
            qml.Hadamard(wires=idx)

    def Rot_layer(self, w):
        """Layer of parametrized qubit rotations around the y axis.
        """
        for idx, element in enumerate(w):
            qml.Rot(element[0], element[1], element[2], wires=idx)


    def entangling_layer(self, nqubits):
        """Layer of CNOTs followed by another shifted layer of CNOT.
        """
        # In other words it should apply something like :
        # CNOT  CNOT  CNOT  CNOT...  CNOT
        #   CNOT  CNOT  CNOT...  CNOT
        for i in range(0, nqubits - 1, 2):  # Loop over even indices: i=0,2,...N-2
            qml.CNOT(wires=[i, i + 1])
        for i in range(1, nqubits - 1, 2):  # Loop over odd indices:  i=1,3,...N-3
            qml.CNOT(wires=[i, i + 1])

    @qml.qnode(dev, interface="torch")
    def quantum_net(self, q_input_features, q_weights_flat):
        """
        The variational quantum circuit.
        """

        # Reshape weights
        q_weights = q_weights_flat.reshape(self.args.q_depth, self.args.n_qubits, 3)

        # Start from state |+> , unbiased w.r.t. |0> and |1>
        # Amplitude encoding
        qml.QubitStateVector(q_input_features, wires=list(range(self.args.n_qubits)))
        
        # Sequence of trainable variational layers
        for k in range(self.args.q_depth):
            self.entangling_layer(self.args.n_qubits)
            self.Rot_layer(q_weights[k])

        # Expectation values in the Z basis
        exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(self.args.target_class)]
        return tuple(exp_vals)





