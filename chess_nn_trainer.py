import logging
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from chess_nn import Chess_NN
from chess_nn_trainer_data import Chess_NN_Trainer_Data

class Chess_NN_Trainer():

    def __init__(self, data_path):
        self.model = Chess_NN()
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.model.net.to(device)
        self.dataset = Chess_NN_Trainer_Data(data_path)

    def train(self, first = False, batch_size = 1, train_size = 0.95, num_epochs = 10, learning_rate = 0.001):

      #if not first_training:
      #  self.model = Chess_NN.load_model()
      self.model.train()
      train_size = int(train_size * len(self.dataset))
      val_size = len(self.dataset) - train_size
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      criterion = torch.nn.MSELoss()
      optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

      train_dataset, val_dataset = torch.utils.data.random_split(self.dataset, [train_size, val_size])

      train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
      val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

      for epoch in range(num_epochs):
        logging.info(f"Epoch: {epoch}")
        self.model.train()
        running_loss = 0.0
        nr_items = len(train_loader)
        logging.info(f"Number Items: {nr_items}")
        index = 0
        for inputs, labels in train_loader:
          index += 1
          # Transfer inputs and labels to the GPU if available
          inputs = inputs.to(device).float()
          labels = labels.to(device).float()
          self.model = self.model.to(device)

          # Forward pass
          outputs = self.model(inputs)

          # Compute the loss
          loss = criterion(outputs, labels)

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          running_loss += loss.item()

          if index % 10000 == 0:
             logging.info(f"Index: {index}")
             logging.info(f"Outputs: {outputs}")
             logging.info(f"Labels: {labels}")
             logging.info(f"Loss: {loss}")
             logging.info(f"Running Loss: {running_loss} {running_loss/index} {math.sqrt(running_loss/(index))}")

        logging.info('[%d, %5d] loss: %.3f' % (epoch + 1, nr_items, running_loss / nr_items, math.sqrt(running_loss / (nr_items))))

        total_loss = 0.0
        nr_items = len(val_loader)
        logging.info(f"Number Items: {nr_items}")
        for inputs, labels in val_loader:
          inputs = inputs.to(device)
          labels = labels.to(device)
          outputs = self.model(inputs)
          loss = criterion(outputs, labels)
          total_loss += loss

        logging.info('Average error of the model on the {} tactics positions is {}'.format(nr_items, total_loss/nr_items, math.sqrt(total_loss / (nr_items))))


      #self.model.save_model()
