import pennylane as qml


import numpy as np 
import torch 

from quant_architecture.quant_arc_interface import * 

class dvqc(quant_arc_interface):
    if torch.cuda.is_available():
        dev1 = qml.device('qulacs.simulator', gpu=True, wires=assign_device())
        dev2 = qml.device('qulacs.simulator', gpu=True, wires=4)         
        print("### dev is gpu")
    else:
        dev1 = qml.device("default.qubit", wires=assign_device())
        dev2 = qml.device("default.qubit", wires=4)
        
        print("### dev is qubit")
    args.quant_architecture = "dvqc"
    args.n_qubits = assign_device()                # Number of qubits
    
    if args.data_name == 'wine':
        args.q_depth = 2

    def __init__(self, args):
        self.args = args 
        self.second_qubits = 4
       

       
        
    def H_layer(self, nqubits):
        """Layer of single-qubit Hadamard gates.
        """
        for idx in range(nqubits):
            qml.Hadamard(wires=idx)

    def RY_layer(self, w):
        """Layer of parametrized qubit rotations around the y axis.
        """
        for idx, element in enumerate(w):
            qml.RY(element, wires=idx)

    def __repr__(self):
        return "dvqc"


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

    @qml.qnode(dev1, interface="torch")
    def quantum_net_1(self, q_input_features, q_weights_flat):
        """
        The variational quantum circuit.
        """

        # Reshape weights
        q_weights = q_weights_flat.reshape(self.args.q_depth, self.args.n_qubits)

        # Start from state |+> , unbiased w.r.t. |0> and |1>
        self.H_layer(self.args.n_qubits)

        # Embed features in the quantum node
        self.RY_layer(q_input_features)

        # Sequence of trainable variational layers
        for k in range(self.args.q_depth):
            self.entangling_layer(self.args.n_qubits)
            self.RY_layer(q_weights[k])

        # Expectation values in the Z basis
        exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(self.second_qubits)]
        return tuple(exp_vals)

    @qml.qnode(dev2, interface="torch")
    def quantum_net_2(self, q_input_features, q_weights_flat):
        """
        The variational quantum circuit.
        """

        # Reshape weights
        q_weights = q_weights_flat.reshape(self.args.q_depth, self.second_qubits)

        # Start from state |+> , unbiased w.r.t. |0> and |1>
        self.H_layer(self.second_qubits)

        # Embed features in the quantum node
        self.RY_layer(q_input_features)

        # Sequence of trainable variational layers
        for k in range(self.args.q_depth):
            self.entangling_layer(self.second_qubits)
            self.RY_layer(q_weights[k])

        # Expectation values in the Z basis
        exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(self.args.target_class)]
        return tuple(exp_vals)