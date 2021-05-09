import torch.nn as nn 
import torch 
import numpy as np 

class apc_net(nn.Module):
    """
    Torch module implementing the *dressed* quantum net.
    """

    def __init__(self, args, quant_arc_interface):
        """
        Definition of the *dressed* layout.
        """

        super().__init__()
        self.args = args 

        self.q_params = nn.Parameter(self.args.q_delta * torch.randn(self.args.q_depth * self.args.n_qubits * 3))
        self.qai = quant_arc_interface


    def forward(self, input_features):
        """
        Defining how tensors are supposed to move through the *dressed* quantum
        net.
        """

        # obtain the input features for the quantum circuit
        q_in = input_features 
        
        # Apply the quantum circuit to each element of the batch and append to q_out
        q_out = torch.Tensor(0, self.args.target_class)
        q_out = q_out.to(self.args.device)
        for elem in q_in:
            elem = elem / torch.clamp(torch.sqrt(torch.sum(elem ** 2)), min = 1e-9)
            q_out_elem = self.qai.quantum_net(self.qai, elem, self.q_params).float().unsqueeze(0).to(q_out)
            q_out = torch.cat((q_out, q_out_elem))

        # return the two-dimensional prediction from the postprocessing layer
        return q_out