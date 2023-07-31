import torch
import torch.nn as nn
import logging

from chess_nn import Chess_NN

class Chess_HalfKP(Chess_NN):
#
# Define a PyTorch net that resembles the Stockfish HalfKP net
#
    def init(self, input_size):
        second_level_size = max(512, input_size*2)
        third_level_size = second_level_size/16
        # The first linear layer takes the 26-int32 board state and
        # produces a 256-dimensional representation.
        self.lin1 = torch.nn.Linear(input_size, second_level_size)

        # The first clipped ReLU layer clips the output of the
        # linear layer to a range of 0 to 1.
        self.clip1 = lambda x: torch.clamp(x, min=0, max=1)

        # The second linear layer takes the 256-dimensional
        # representation and produces a 32-dimensional representation.
        self.lin2 = torch.nn.Linear(second_level_size, 32)

        # The second clipped ReLU layer clips the output of the
        # linear layer to a range of 0 to 1.
        self.clip2 = lambda x: torch.clamp(x, min=0, max=1)

        # The third linear layer takes the 32-dimensional
        # representation and produces a 1-dimensional representation.
        self.lin3 = torch.nn.Linear(32, 1)

        # The fourth linear layer takes the 1-dimensional
        # representation and produces a 1-dimensional representation
        # with a range of -15 to 15.
        self.lin4 = torch.nn.Linear(1, 1)
        self.lin4.weight.data = (2 * torch.randn(1, 1) - 1) * 15

        # Initialize the network parameters to zeros
        self.init_weights()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        size = (26,)
        self.sample_input = torch.randint(0, 100, size=size, dtype=torch.int32).to(device).float()
        self.onnx_counter = 0

    def __init__(self):
        super(Chess_NN, self).__init__()
        self.init(26)

    def init_weights(self):
        for layer in [self.lin1, self.lin2, self.lin3, self.lin4]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        #logging.debug(f"Input Net: {x} {x.shape} {x.dtype}")
        x = self.lin1(x)
        x = self.clip1(x)
        x = self.lin2(x)
        x = self.clip2(x)
        x = self.lin3(x)
        x = self.lin4(x)

        # Clip the output to be between -15.0 and 15.0
        x = self.clamp_output(x)
        #logging.debug(f"Output Net: {x}")

        return x.view(-1)
    
    def get_name(self):
        return "HALFKP_V01"