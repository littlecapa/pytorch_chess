import logging
from torch.utils.data import DataLoader
from chess_nn_trainer_data import Chess_NN_Trainer_Data


FILEPATH = "G:/Meine Ablage/data/nn_chess/"
FILE_PREFIX ="commentated_li_"
FILE_RANGE = 100

def setup_logging():
    logging.basicConfig(
        filename='app_stat.log',  # Change this to your desired log file path
        level=logging.INFO,  # Change the log level as needed (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format='%(asctime)s [%(levelname)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def check_env():
    logging.info(f"FILEPATH: {FILEPATH}")
    logging.info(f"FILE_RANGE: {FILE_RANGE}")
    logging.info(f"FILE_PREFIX: {FILE_PREFIX}")

def main():
    setup_logging()
    logging.info('Starting the program')

    check_env()
    min = 0
    max = 0
    over_10 = 0
    under_10 = 0
    over_1 = 0
    under_1 = 0
    over_5 = 0
    under_5 = 0
    total = 0
    output_file = FILEPATH + "eval_list.csv"
    outfile = open(output_file, "w")
    
    for i in range(FILE_RANGE):
        input_file = FILEPATH + FILE_PREFIX + str(i) + ".csv"

        dataset = Chess_NN_Trainer_Data(input_file, scale_eval = False)
        dataloader = DataLoader(dataset, batch_size = 1, shuffle=False)
        for _, batch_label in dataloader:
            batch_label = batch_label.item()
            outfile.write(str(batch_label) + "\n")
            total += 1
            if batch_label > max:
                max = batch_label
            elif batch_label < min:
                min = batch_label
            if batch_label < -10:
                under_10 += 1
            if batch_label > 10:
                over_10 += 1
            if batch_label < -5:
                under_5 += 1
            if batch_label > 5:
                over_5 += 1
            if batch_label < -1:
                under_1 += 1
            if batch_label > 1:
                over_1 += 1

        logging.info(f"Total: {total}, Min: {min}, Max: {max}, Under_10: {under_10*100/total}%, Over_10: {over_10*100/total}%")
        logging.info(f"Between -5 and 5: {(total-under_5-over_5)*100/total}%, Between -1 and 1: {(total-under_1-over_1)*100/total}%")
        outfile.flush()
    outfile.close()
    logging.info('Program execution completed')

# Call the main function if the script is executed directly
if __name__ == "__main__":
    main()
