from utils import load_images_from_folder as loader
from utils import load_data
import cv2
import numpy as np
import random
from sklearn.model_selection import train_test_split
import models_handling
import utils
import argparse
import pickle
import visualizations

def main():
    data = utils.data_prepare()

    # PARAMETERS INITIALIZATION.
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", help="batch size for training", default=64)
    parser.add_argument("--output_size", help="output size, number of classes",
                        default=2)
    parser.add_argument("--learning_rate", help="learning rate for network training", default=0.0005)
    parser.add_argument("--epochs", help="number of epochs for training", default=60)
    parser.add_argument("--optimizer", help="optimizer for training", default='adam')
    parser.add_argument("--verbose", help="verbosity", default=1)
    parser.add_argument("--flag_visualizations", help="flag for presenting plots of the training process", default=1)
    parser.add_argument("--save_path", help="save model path", default="checkpoints")

    args = parser.parse_args()
    args_dict = vars(args)
    with open('temp_data/arguments', 'wb') as file_pi:
        pickle.dump(args_dict, file_pi)

    history = models_handling.model_execution(data=data,
                                              output_size=args.output_size,
                                              batch_size=args.batch_size,
                                              optimizer=args.optimizer,
                                              epochs=args.epochs,
                                              lr=args.learning_rate,
                                              verbose=args.verbose,
                                              save_path=args.save_path)

    accuracy = list(history.history['accuracy'])

    if args.flag_visualizations:
        # visualizations.plot_accuracy(accuracy, network)
        visualizations.plot_learning_curves(history)


if __name__ == "__main__":
    main()
