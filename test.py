import tensorflow as tf
import utils
import argparse
import pickle


def test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="path for model .pb file", type=str, default="checkpoints5/saved_model.pb")
    parser.add_argument("--data_path", help="path for the data to load", type=str, default="data")
    parser.add_argument("--verbose", help="verbosity", default=1)

    args = parser.parse_args()
    args = vars(args)

    # Load saved model
    model = tf.keras.models.load_model(
        args["model_path"], custom_objects=None, compile=True, options=None
    )

    x_test, y_test = utils.test_data_load(args["data_path"], False)

    utils.print_evaluation(model, x_test, y_test, verbose=args["verbose"])


if __name__ == "__main__":
    test()
