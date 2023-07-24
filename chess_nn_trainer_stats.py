import logging
import datetime
import math

class Chess_NN_Trainer_Stats():

    def __init__(self, file_path):
        current_datetime = self.datetime2str()  
        filename_stats = file_path + "stats_" + current_datetime + ".csv"
        self.outfile_stats = open(filename_stats, "w")
        logging.debug(f"Stats File created {filename_stats}")
        filename_info = file_path + "info_" + current_datetime + ".txt"
        self.outfile_info = open(filename_info, "w")
        logging.debug(f"Info File created {filename_info}")

    def __del__(self):
        self.flush()
        self.outfile_info.close()
        self.outfile_stats.close()

    def flush(self):
        self.outfile_info.flush()
        self.outfile_stats.flush()
    
    def datetime2str(self):
        # Get the current datetime
        current_datetime = datetime.datetime.now()

        # Convert it to a string with a specific format
        # For example, to display the date and time in ISO 8601 format:
        return current_datetime.strftime('%Y_%m_%d_%H_%M_%S')

    def write_info(self, parameter, value):
        self.outfile_info.write(f"{datetime.datetime.now()};{parameter};{value}\n")

    def write_net_infos(self, model):
        for name, param in model.state_dict().items():
            self.write_info(name, param.shape)

    def start_file(self, filename, nr_items, nr_items_val):
        self.latest_filename = filename
        self.latest_nr_items = nr_items
        self.latest_nr_items_val = nr_items_val
        self.outfile_stats.write(f"{datetime.datetime.now()};{filename};{nr_items};Start File;;;;\n")
        self.flush()

    def start_epoch(self, epoch):
        self.latest_epoch = epoch
        self.outfile_stats.write(f"{datetime.datetime.now()};{self.latest_filename};{self.latest_nr_items};Start Epoch;{str(epoch)};;;\n")
    
    def end_epoch(self, total_loss):
        self.outfile_stats.write(f"{datetime.datetime.now()};{self.latest_filename};{self.latest_nr_items};End Epoch;{str(self.latest_epoch)};{total_loss};{total_loss/self.latest_nr_items};{math.sqrt(total_loss/self.latest_nr_items)}\n")
        self.flush()

    def end_validation(self, running_loss):
        self.outfile_stats.write(f"{datetime.datetime.now()};{self.latest_filename};{self.latest_nr_items_val};End Validation;{str(self.latest_epoch)};{running_loss};{running_loss/self.latest_nr_items_val};{math.sqrt(running_loss/self.latest_nr_items_val)}\n")
        self.flush()

    def end_file(self, total_loss, num_epochs):
        self.outfile_stats.write(f"{datetime.datetime.now()};{self.latest_filename};{self.latest_nr_items};End File;;{total_loss};{total_loss/(self.latest_nr_items*num_epochs)};{math.sqrt(total_loss/(self.latest_nr_items*num_epochs))}\n")
        self.flush()
