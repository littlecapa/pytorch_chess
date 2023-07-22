import torch
import torch.nn as nn
import logging

from chess_nn import Chess_NN

class Chess_HalfKP(Chess_NN):
#
# Define a PyTorch net that resembles the Stockfish HalfKP net
#
    def __init__(self):
        super(Chess_HalfKP, self).__init__()

        # The first linear layer takes the 26-int32 board state and
        # produces a 256-dimensional representation.
        self.lin1 = torch.nn.Linear(26, 512)

        # The first clipped ReLU layer clips the output of the
        # linear layer to a range of 0 to 1.
        self.clip1 = lambda x: torch.clamp(x, min=0, max=1)

        # The second linear layer takes the 256-dimensional
        # representation and produces a 32-dimensional representation.
        self.lin2 = torch.nn.Linear(512, 32)

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

    def forward(self, x):
        logging.debug(f"Input Net: {x}")
        #x = x.to(torch.float32)  # Convert input to float32
        x = self.lin1(x)
        x = self.clip1(x)
        x = self.lin2(x)
        x = self.clip2(x)
        x = self.lin3(x)
        x = self.lin4(x)

        # Clip the output to be between -15.0 and 15.0
        x = self.clamp_output(x)
        logging.debug(f"Output Net: {x}")

        return x.view(-1)