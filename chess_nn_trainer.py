import logging
import math
import torch
import io
from chess_halfkp_test import Chess_HalfKP
#from chess_halfkp import Chess_HalfKP
from chess_nn_trainer_data import Chess_NN_Trainer_Data
from chess_nn_trainer_test_data import Chess_NN_Trainer_Test_Data
from chess_nn_trainer_stats import Chess_NN_Trainer_Stats
from math import sqrt
from loss_function import eval_loss_function

class Chess_NN_Trainer():

  INIT_HASH = 0

  def __init__(self, stats_path, baseline_test_data_path, first_training = False, filename = "chess.h5", learning_rate = 1e-2):
    self.stats = Chess_NN_Trainer_Stats(stats_path)
    test_dataset = Chess_NN_Trainer_Test_Data(baseline_test_data_path)
    self.test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
    self.filename = filename
    self.learning_rate = learning_rate
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.init_components()
    self.hash = self.INIT_HASH
    if not first_training:
      self.resume()
    else:
       self.do_baseline_testing()

  def init_components(self):
    self.machine = Chess_HalfKP()
    self.machine.to(self.device)
    self.stats.write_net_infos(self.machine)
    self.optimizer = torch.optim.Adam(self.machine.parameters(), self.learning_rate, weight_decay=1e-5)
    self.criterion = torch.nn.MSELoss(reduction='mean')
    torch.backends.cudnn.enabled = True

  def rest(self):
    torch.save(self.machine.state_dict(), self.filename)
    new_hash = self.machine.get_hash_value()
    self.hash = new_hash
    logging.info(f"Resting Hash: {self.hash}")
    self.machine = None

  def resume(self):
    self.init_components()
    self.machine.load_state_dict(torch.load(self.filename))
    new_hash = self.machine.get_hash_value()
    logging.info(f"Resuming Hash: {new_hash}")
    self.hash = new_hash
    self.do_baseline_testing()
  
  def run_machine(self, loader, training = False, base_line = False):
    if training:
      self.machine.train()
    else:
       self.machine.eval()
    running_loss = 0.0
    index = 0
    limit = len(loader)/100
    for x, y in loader:
      index += 1
      x = x.to(self.device).float()
      y = y.to(self.device).float()
      logging.debug(f"X: {x}")
      outputs = self.machine.forward(x)
      #loss = self.criterion(outputs, y)
      loss = eval_loss_function(outputs, y, self.device, self.stats)
      if base_line or index > limit:
        self.stats.write_test_results(index, "", outputs, y)
        limit *= 2
      if training:
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
      running_loss += loss.item()
    return running_loss        

  def do_training(self, data_path, batch_size = 1, train_size = 0.95, num_epochs = 10, shuffle = False):

    self.dataset = Chess_NN_Trainer_Data(data_path)
    self.stats.start_file(self.data_path, train_size, 1-train_size)
    train_size = int(train_size * len(self.dataset))
    val_size = len(self.dataset) - train_size        
    train_dataset, val_dataset = torch.utils.data.random_split(self.dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    sum_loss = 0
    for epoch in range(num_epochs):
        self.stats.start_epoch(epoch)
        running_loss = self.run_machine(train_loader, training = True)
        sum_loss += running_loss
        logging.debug(f'Training Results! Epoch:{epoch}, Running Loss:{int(running_loss)}, Items: {int((len(train_dataset)))}, Avg: {round(running_loss/(len(train_dataset)),2)} {round(sqrt(running_loss/(len(train_dataset))),2)}')
        self.do_validation(val_loader)
        self.do_baseline_testing()
        self.stats.end_epoch(running_loss)
    self.rest()
    self.stats.end_file(sum_loss, num_epochs)

  def do_validation(self, val_loader):
    #with torch.no_grad():
    running_loss = self.run_machine(val_loader, training = False)
    logging.debug(f'Validation Results! Running Loss:{running_loss}, Items: {len(val_loader)}, Avg: {running_loss/len(val_loader)}')
    self.stats.end_validation(running_loss)

  def do_baseline_testing(self):
    self.stats.start_testing()
    #with torch.no_grad():
    running_loss = self.run_machine(self.test_data_loader, training = False, base_line = True)
    logging.debug(f'Results Baseline Testing! Running Loss:{running_loss}, Items: {len(self.test_data_loader)}, Avg: {running_loss/len(self.test_data_loader)}')
    self.stats.end_testing()