import models_handling
import utils
import argparse
import pickle
import visualizations
import tensorflow as tf


def main():


    # PARAMETERS INITIALIZATION.
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", help="batch size for training", default=64)
    parser.add_argument("--output_size", help="output size, number of classes",
                        default=2)
    parser.add_argument("--learning_rate", help="learning rate for network training", default=0.0005)
    parser.add_argument("--epochs", help="number of epochs for training", type=int, default=60)
    parser.add_argument("--optimizer", help="optimizer for training", default='adam')
    parser.add_argument("--verbose", help="verbosity", default=1)
    parser.add_argument("--flag_visualizations", help="flag for presenting plots of the training process", default=1)
    parser.add_argument("--save_path", help="save model path", default="checkpoints")
    parser.add_argument("--seed", help="seed for reproducible train/test distribution", default=123)
    parser.add_argument("--model_summary", help="plot the model summary", default=False)

    args = parser.parse_args()
    args_dict = vars(args)
    with open('temp_data/arguments', 'wb') as file_pi:
        pickle.dump(args_dict, file_pi)

    data = utils.data_prepare(seed=args.seed)
    x_train, x_val, x_test, y_train, y_val, y_test = data

    model, history = models_handling.model_training(data=data,
                                                    output_size=args.output_size,
                                                    batch_size=args.batch_size,
                                                    optimizer=args.optimizer,
                                                    epochs=args.epochs,
                                                    lr=args.learning_rate,
                                                    verbose=args.verbose,
                                                    )

    if args.model_summary:
        print(model.summary())

    utils.print_evaluation(model, x_test, y_test, verbose=args.verbose)
    tf.keras.models.save_model(model, args.save_path)

    if args.flag_visualizations:
        visualizations.plot_learning_curves(history)


if __name__ == "__main__":
    main()
