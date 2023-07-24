import logging
import math
import torch
#import torch.nn as nn
#from torch.utils.data import DataLoader, Dataset
from chess_nn import Chess_NN
from chess_halfkp import Chess_HalfKP
from chess_nn_trainer_data import Chess_NN_Trainer_Data
from chess_nn_trainer_stats import Chess_NN_Trainer_Stats

class Chess_NN_Trainer():

    def __init__(self, FILEPATH, stats, data_path, use_halfkp = False):
        self.stats = stats
        if use_halfkp:
          self.model = Chess_HalfKP()
          self.stats.write_info("Model", "Chess_HalfK")
        else:
          self.model = Chess_NN()
          self.stats.write_info("Model", "Chess_NN")
        self.stats.write_net_infos(self.model)
        self.data_path = data_path
        self.dataset = Chess_NN_Trainer_Data(data_path)
        self.filepath = FILEPATH
        self.use_halfkp = use_halfkp
        self.net_hash_value = None

    def train(self, first_training = False, batch_size = 1, train_size = 0.95, num_epochs = 10, learning_rate = 0.001, shuffle = False, use_mse = True):

      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      self.stats.write_info("Device", device)
      self.stats.write_info("Batch Size", batch_size)
      self.stats.write_info("Train Size", train_size)
      self.stats.write_info("Learning Rate", learning_rate)
      self.stats.write_info("Shuffle", shuffle)
      if not first_training:
        if self.use_halfkp:
          self.model = Chess_HalfKP.load_model(self.filepath)
        else:
          self.model = Chess_NN.load_model(self.filepath)
        new_net_hash_value = self.model.get_hash_value()
        logging.info(f"New Hash Value: {new_net_hash_value}")
        if self.net_hash_value is not None:
          logging.info(f"Old Hash Value: {self.net_hash_value}")
          if self.net_hash_value != new_net_hash_value:
            raise ValueError(f"Invalid Net Hash Value! Expected:{self.net_hash_value}, got {new_net_hash_value}")
          
      self.model.train()
      self.model = self.model.to(device)
      train_size = int(train_size * len(self.dataset))
      val_size = len(self.dataset) - train_size
      
      if use_mse:
        criterion = torch.nn.MSELoss()
        self.stats.write_info("Criterion", "torch.nn.MSELoss")
      else:
        criterion = torch.nn.L1Loss()  # Use L1Loss for MAE
        self.stats.write_info("Criterion", "torch.nn.L1Losss")
      optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

      train_dataset, val_dataset = torch.utils.data.random_split(self.dataset, [train_size, val_size])

      train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
      val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)

      self.stats.start_file(self.data_path, train_size, val_size)

      sum_loss = 0

      for epoch in range(num_epochs):
        self.stats.start_epoch(epoch)
        self.model.train()
        running_loss = 0.0
        nr_items = len(train_loader)
        index = 0
        for inputs, labels in train_loader:
          index += 1
          # Transfer inputs and labels to the GPU if available
          inputs = inputs.to(device).float()
          labels = labels.to(device).float()
          #self.model = self.model.to(device)

          # Forward pass
          outputs = self.model(inputs)

          # Compute the loss
          loss = criterion(outputs, labels)

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          running_loss += loss.item()

          if index % (40960/batch_size) == 0:
             print(f"Index {index}")
             logging.debug(f"Index: {index}")
             logging.debug(f"Outputs: {outputs}")
             logging.debug(f"Labels: {labels}")
             logging.debug(f"Loss: {loss} {math.sqrt(loss)} {abs(outputs-labels)}")
             logging.debug(f"Running Loss: {running_loss/index} {math.sqrt(running_loss/index)}")
        sum_loss += running_loss 

        self.stats.end_epoch(running_loss)
        total_loss = 0.0
        nr_items = len(val_loader)
        for inputs, labels in val_loader:
          inputs = inputs.to(device).float()
          labels = labels.to(device).float()
          outputs = self.model(inputs)
          loss = criterion(outputs, labels)
          total_loss += loss
        self.stats.end_validation(total_loss)

      self.stats.end_file(sum_loss, num_epochs)
      self.net_hash_value = self.model.get_hash_value()
      logging.info(f"New Hash Value: {self.net_hash_value}")
      self.model.save_model(self.filepath)
