import torch.nn as nn 
import torch 
import numpy as np 



class dvqc_net(nn.Module):

	def __init__(self, q_net_1, q_net_2):
		super(dvqc_net, self).__init__()
		self.q_net_1 = q_net_1
		self.q_net_2 = q_net_2

	def forward(self, x):
		x = self.q_net_1(x)
		x = self.q_net_2(x)

		return x





class QNet_1(nn.Module):
    """
    Torch module implementing the *dressed* quantum net.
    """

    def __init__(self, quant_arc_interface):
        """
        Definition of the *dressed* layout.
        """

        super().__init__()
        self.args = quant_arc_interface.args 
        self.q_params = nn.Parameter(self.args.q_delta * torch.randn(self.args.q_depth * self.args.n_qubits))
        self.qai = quant_arc_interface

    def forward(self, input_features):
        """
        Defining how tensors are supposed to move through the *dressed* quantum
        net.
        """

        # obtain the input features for the quantum circuit
        # by reducing the feature dimension from 512 to 4
        q_in = torch.tanh(input_features) * np.pi / 2.0

        # Apply the quantum circuit to each element of the batch and append to q_out
        q_out = torch.Tensor(0, self.qai.second_qubits)
        q_out = q_out.to(self.args.device)
        for elem in q_in:
            q_out_elem = self.qai.quantum_net_1(self.qai, elem, self.q_params).float().unsqueeze(0).to(self.args.device)
            # print('###q_out is cuda', q_out.is_cuda)
            q_out = torch.cat((q_out, q_out_elem))

        # return the two-dimensional prediction from the postprocessing layer
        return q_out


class QNet_2(nn.Module):
    """
    Torch module implementing the *dressed* quantum net.
    """

    def __init__(self, quant_arc_interface):
        """
        Definition of the *dressed* layout.
        """

        super().__init__()
        self.args = quant_arc_interface.args
        self.q_params = nn.Parameter(self.args.q_delta * torch.randn(self.args.q_depth * quant_arc_interface.second_qubits))
        self.qai = quant_arc_interface

    def forward(self, input_features):
        """
        Defining how tensors are supposed to move through the *dressed* quantum
        net.
        """

        # obtain the input features for the quantum circuit
        # by reducing the feature dimension from 512 to 4
        q_in = torch.tanh(input_features) * np.pi / 2.0

        # Apply the quantum circuit to each element of the batch and append to q_out
        q_out = torch.Tensor(0, self.args.target_class)
        q_out = q_out.to(self.args.device)
        for elem in q_in:
            q_out_elem = self.qai.quantum_net_2(self.qai, elem, self.q_params).float().unsqueeze(0).to(self.args.device)
            q_out = torch.cat((q_out, q_out_elem))

        # return the two-dimensional prediction from the postprocessing layer
        return q_out


