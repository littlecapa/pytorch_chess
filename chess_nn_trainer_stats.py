import logging
import datetime
import math
import random

class Chess_NN_Trainer_Stats():

    def __init__(self, file_path):
        method = 'a'
        current_datetime = self.datetime2str()  
        filename_stats = file_path + "stats_" + current_datetime + ".csv"
        self.outfile_stats = open(filename_stats, method)
        logging.debug(f"Stats File created {filename_stats}")
        filename_info = file_path + "info_" + current_datetime + ".txt"
        self.outfile_info = open(filename_info, method)
        logging.debug(f"Info File created {filename_info}")
        filename_tests = file_path + "tests_" + current_datetime + ".txt"
        self.outfile_tests = open(filename_tests, method)
        logging.debug(f"Info File created {filename_tests}")
        self.latest_filename = ""
        self.loss_calls = random.randint(0, 9999)

    def __del__(self):
        self.flush()
        self.outfile_info.close()
        self.outfile_stats.close()
        self.outfile_tests.close()

    def flush(self):
        self.outfile_info.flush()
        self.outfile_stats.flush()
        self.outfile_tests.flush()
    
    def datetime2str(self):
        # Get the current datetime
        current_datetime = datetime.datetime.now()

        # Convert it to a string with a specific format
        # For example, to display the date and time in ISO 8601 format:
        return current_datetime.strftime('%Y_%m_%d_%H_%M_%S')

    def start_testing(self, info = "Start"):
        self.error = 0
        self.nr_tests = 0
        self.outfile_tests.write(f"{datetime.datetime.now()};{info} - {self.latest_filename};;;\n")
        self.latest_filename
        self.start_tests = True

    def write_test_results(self, test_id, test_info, value, expected_value):
        if self.start_tests:
            self.start_tests = False
            self.first_value = value
            self.bad_testing = True
        if self.first_value != value:
            self.bad_testing = False
        self.nr_tests += 1
        v = round(value.item(),2)
        e = round(expected_value.item(),2)
        logging.info(f"Result: {v} {e}")
        self.error += abs(v-e)
        self.outfile_tests.write(f"{datetime.datetime.now()};test result;{test_id};{v};{e};{abs(v-e)}\n")
        if test_info != "":
            pass
            #self.outfile_tests.write(f"{datetime.datetime.now()};{test_info};;;\n")
        if test_id > 1000:
            self.flush()

    def end_testing(self):
        self.outfile_tests.write(f"{datetime.datetime.now()};total results;;{self.nr_tests};{round(self.error,2)};{round(self.error/self.nr_tests, 2)}\n")
        self.error = 0
        self.nr_tests = 0
        self.flush()
        if self.bad_testing:
            raise ValueError("Bad Testing")

    def write_info(self, parameter, value):
        self.outfile_info.write(f"{datetime.datetime.now()};{parameter};{value}\n")

    def write_net_infos(self, model):
        for name, param in model.state_dict().items():
            self.write_info(name, param.shape)

    def write_loss_function_infos(self, out_tensor, label_tensor, loss_tensor):
        self.loss_calls += 1
        if self.loss_calls > 5000:
            self.loss_calls = 0
            for out, label, loss in zip(out_tensor, label_tensor, loss_tensor):
                self.write_info("Out", out)
                self.write_info("Label", label)
                self.write_info("Loss", loss)

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
        self.outfile_stats.write(f"{datetime.datetime.now()};{self.latest_filename};{self.latest_nr_items};End Epoch;{str(self.latest_epoch)};{total_loss};{total_loss/self.latest_nr_items}\n")
        self.flush()

    def end_validation(self, running_loss):
        self.outfile_stats.write(f"{datetime.datetime.now()};{self.latest_filename};{self.latest_nr_items_val};End Validation;{str(self.latest_epoch)};{running_loss};{running_loss/self.latest_nr_items_val}\n")
        self.flush()

    def end_file(self, total_loss, num_epochs):
        self.outfile_stats.write(f"{datetime.datetime.now()};{self.latest_filename};{self.latest_nr_items};End File;;{total_loss};{total_loss/(self.latest_nr_items*num_epochs)};{math.sqrt(total_loss/(self.latest_nr_items*num_epochs))}\n")
        self.flush()
