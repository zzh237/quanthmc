
# input_size = 784
# hidden_sizes = [128, 64]
# output_size = 10

# model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
#                       nn.ReLU(),
#                       nn.Linear(hidden_sizes[0], hidden_sizes[1]),
#                       nn.ReLU(),
#                       nn.Linear(hidden_sizes[1], output_size),
#                       nn.LogSoftmax(dim=1))



import torch.nn as nn
class MLP(nn.Module):
    def __init__(self, input_dim, width, depth, output_dim):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.width = width
        self.depth = depth
        
        if depth == 0:
            self.block = nn.Linear(input_dim, output_dim, bias=True)

        else:
            layers = [nn.Linear(input_dim, width), nn.ReLU()]
            for i in range(depth - 1):
                layers.append(nn.Linear(width, width))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(width, output_dim))
            self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)