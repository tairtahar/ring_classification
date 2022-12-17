import tensorflow as tf
from lenet import LeNetFCBN3


def model_selection(input_shape, output_size):
    """This function receives current chosen notwork and parameters to be used for network initialization, and returns
    an instance of the chosen network"""
    print("Chosen network is LeNet with Batchnorm on first + second convolution layers + 2 fully connected layers")
    lenet_model = LeNetFCBN3(input_shape=input_shape, output_size=output_size)

    return lenet_model


def model_execution(data, output_size, batch_size, optimizer, epochs, lr, verbose, save_path):
    """This function executes all the steps: model creation, compilation, training, and evaluation"""
    x_train, x_val, x_test, y_train, y_val, y_test = data
    input_shape = x_train.shape[1:]
    lenet_model = model_selection(input_shape, output_size)
    lenet_model.model_compilation(optimizer=optimizer, learning_rate=lr)
    history = lenet_model.train(x_train, y_train, x_val, y_val,
                                batch_size=batch_size,
                                epochs=epochs,
                                verbose=verbose)
    lenet_model.print_evaluation(x_test, y_test, verbose=verbose)
    tf.keras.models.save_model(lenet_model, save_path)
    return history

