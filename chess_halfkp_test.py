import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from chess_nn import Chess_NN

class Chess_HalfKP(Chess_NN):
#
# Define a PyTorch net that resembles the Stockfish HalfKP net
#
    def init_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3]:
            # nn.init.xavier_uniform_(layer.weight)
            # Initialize the weights with values from a normal distribution
            #nn.init.normal_(layer.weight, mean=0.0, std=0.01)
            #nn.init.zeros_(layer.bias)
            nn.init.normal_(layer.weight, mean=0.0, std=1.0)
            nn.init.normal_(layer.bias, mean=0.0, std=1.0)

    def define_net(self):
        # Define the layers of the network
        self.fc1 = nn.Linear(26, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)

        # Initialize the network parameters
        self.init_weights()

    def __init__(self, max_eval_value = 15.0):
        super(Chess_HalfKP, self).__init__(max_eval_value)
        self.define_net()
        self.init_internals(max_eval_value)

    def process_forward(self, x):
        # Pass the input through the layers of the network
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
    
    def get_name(self):
        return "HALFKP_TEST_V01"
    
    CLAMP_MAX = 15.0

    def clamp_output(self, x):
        for value in x:
            self.check_stats(value.item())
        # Clamp values between -MAX and + MAX
        clamped_tensor = torch.clamp(x, min=-self.CLAMP_MAX, max=self.CLAMP_MAX)
        logging.debug(f"Clamping {x} => {clamped_tensor}")
    
        return x
