import pennylane as qml

from experiments.ind_exp import args
import numpy as np 

def assign_device():
    feature_size = 10
    if args.data_name == 'cancer':
        feature_size = 5
    if args.data_name == 'wine':
        feature_size = 13
    if args.data_name == 'iris':
        feature_size = 4
    if args.data_name == 'mnist':
        feature_size = 728
    if args.data_name == 'digits':
        feature_size = 64
    
    
    return feature_size     

class vqc():
    dev = qml.device("default.qubit", wires=assign_device())
    def __init__(self, args):
        self.args = args 
        
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
        exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(self.args.target_class)]
        return tuple(exp_vals)