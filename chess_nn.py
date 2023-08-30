import torch
import torch.nn as nn
import logging
import numpy as np
from nn_hash import nn_Hash

class Chess_NN(nn_Hash):
    def init_internals(self, max_eval_value):
        self.min_eval = 0.0
        self.max_eval = 0.0
        self.nr_evals = 0
        self.max_eval_value = max_eval_value

    def get_eval_stats(self):
        return self.min_eval, self.max_eval, self.nr_evals
    
    def log_eval_stats(self):
        logging.info(f"Min: {self.min_eval}, Max: {self.max_eval}, #{self.nr_evals}")

    def check_stats(self, value):
        self.nr_evals += 1
        if value < self.min_eval:
            self.min_eval = value
        elif value > self.max_eval:
            self.max_eval = value
    
    def __del__(self):
        self.log_eval_stats()

    def create_net(self):
        self.fc1 = nn.Linear(26, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)

        # Initialize the network parameters to zeros
        self.init_weights()
        size = (26,)
        self.sample_input = torch.randint(0, 100, size=size, dtype=torch.int32)
        
    def __init__(self, max_eval_value = 15.):
        super(Chess_NN, self).__init__()
        self.create_net()
        self.init_internals(max_eval_value)
    
    def get_name(self):
        return "NN_V01"

    def init_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def process_forward(self, x):
        self.sample_input = x
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        logging.debug(f"Unclamped Output: {x}")
        
    def forward(self, x):
        logging.debug(f"Input Net: {x}")
        x = self.process_forward(x)
        logging.debug(f"Unclamped Output: {x}")
        x = self.clamp_output(x)
        logging.debug(f"Clamped Output: {x}")
        return x.view(-1)
    
    def clamp_output(self, x):
        device = x.device
        requires_grad = x.requires_grad
        x = torch.tensor([self.clamp_float(x_value) for x_value in x], requires_grad=requires_grad).to(device)
        return x
    
    def clamp_float(self, value,  max = np.finfo(np.float32).max):
        self.check_stats(value)
        clamped_value = (value / max) * self.max_eval_value
        return clamped_value
    