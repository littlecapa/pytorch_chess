import torch
import logging
from chess_nn_trainer_stats import Chess_NN_Trainer_Stats

from chess_nn_trainer import Chess_NN_Trainer

FILEPATH = "G:/Meine Ablage/data/nn_chess/"
#FILEPATH = "C:/Users/littl/Documents/AI/tmp/"
FILE_PREFIX ="commentated_li_"
FILE_RANGE = 1
SMALL_TEST = True

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
    logging.debug('Starting the program')
    stats = Chess_NN_Trainer_Stats(FILEPATH + "stats/")

    check_env(stats)
    first_training = True
    
    for i in range(FILE_RANGE):
        if SMALL_TEST:
            input_file = FILEPATH + "merge.csv"
        else:
            input_file = FILEPATH + FILE_PREFIX + str(i) + ".csv"
        trainer = Chess_NN_Trainer(FILEPATH, stats, input_file, use_halfkp = True)
        trainer.train(first_training = first_training, shuffle = True, batch_size=1024, num_epochs = 10, learning_rate = 0.1)
        first_training = False

    logging.debug('Program execution completed')

# Call the main function if the script is executed directly
if __name__ == "__main__":
    main()
