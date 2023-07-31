import torch
import logging
from chess_nn_trainer_stats import Chess_NN_Trainer_Stats

from chess_nn_trainer import Chess_NN_Trainer

FILEPATH = "G:/Meine Ablage/data/nn_chess/"
#FILEPATH = "C:/Users/littl/Documents/AI/tmp/"
FILE_PREFIX ="commentated_li_"
FILE_RANGE = 2
SMALL_TEST = False

def setup_logging():
    logging.basicConfig(
        filename='app.log',  # Change this to your desired log file path
        level=logging.INFO,  # Change the log level as needed (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format='%(asctime)s [%(levelname)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def check_env(stats):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.debug(f"Torch Device: {device}")
    logging.debug(f"FILEPATH: {FILEPATH}")
    logging.debug(f"FILE_RANGE: {FILE_RANGE}")
    logging.debug(f"FILE_PREFIX: {FILE_PREFIX}")
    stats.write_info("Torch Device", device)
    stats.write_info("FILEPATH", FILEPATH)
    stats.write_info("FILE_RANGE", FILE_RANGE)
    stats.write_info("FILE_PREFIX", FILE_PREFIX)

def main():
    setup_logging()
    first_training = True
    logging.debug('Starting the program')
    input_file_pattern = FILEPATH + "commentated_li_"
    #input_file = FILEPATH + "commentated_wolga_9.csv"
    for i in range(2):
        input_file = input_file_pattern + str(i) + ".csv"
        trainer = Chess_NN_Trainer( stats_path = FILEPATH + "stats/", data_path = input_file, baseline_test_data_path = FILEPATH + "baseline_test_data.csv", first_training = first_training, filename = "chess.h5", learning_rate = 1e-2)
        trainer.do_training(batch_size = 1, train_size = 0.98, num_epochs = 2, shuffle = False)
        first_training = False
    logging.debug('Program execution completed')

# Call the main function if the script is executed directly
if __name__ == "__main__":
    main()
