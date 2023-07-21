import torch
import logging

from chess_nn_trainer import Chess_NN_Trainer

FILEPATH = "G:/Meine Ablage/data/nn_chess/"
FILE_PREFIX ="commentated_li_"
FILE_RANGE = 1

def setup_logging():
    logging.basicConfig(
        filename='app.log',  # Change this to your desired log file path
        level=logging.INFO,  # Change the log level as needed (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format='%(asctime)s [%(levelname)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def check_env():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Torch Device: {device}")
    logging.info(f"FILEPATH: {FILEPATH}")
    logging.info(f"FILE_RANGE: {FILE_RANGE}")
    logging.info(f"FILE_PREFIX: {FILE_PREFIX}")

def main():
    setup_logging()
    logging.info('Starting the program')

    check_env()
    
    for i in range(FILE_RANGE):
        input_file = FILEPATH + FILE_PREFIX + str(i) + ".csv"
        trainer = Chess_NN_Trainer(input_file)
        trainer.train()

    logging.info('Program execution completed')

# Call the main function if the script is executed directly
if __name__ == "__main__":
    main()
