import torch
import torch.nn as nn
import torch.onnx
import logging
from nn_hash import nn_Hash

class Chess_NN(nn_Hash):
    def __init__(self):
        super(Chess_NN, self).__init__()

        self.fc1 = nn.Linear(26, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)

        # Initialize the network parameters to zeros
        self.init_weights()
        self.set2train()
        size = (26,)
        self.sample_input = torch.randint(0, 100, size=size, dtype=torch.int32)
        self.onnx_counter = 0
    
    def get_name(self):
        return "NN_V01"

    def init_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        logging.debug(f"Input Net: {x}")
        self.sample_input = x
        #x = x.to(torch.float32)  # Convert input to float32
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        x = self.clamp_output(x)
        logging.debug(f"Output Net: {x}")
        return x.view(-1)
    
    def clamp_output(self, x):
         # Clip the output to be between -15.0 and 15.0
        return torch.clamp(x, min=-15.0, max=15.0)
    