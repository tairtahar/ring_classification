from lenet import LeNetFCBN3


def model_selection(input_shape, output_size):
    """This function receives current chosen notwork and parameters to be used for network initialization, and returns
    an instance of the chosen network"""
    print("Chosen network is LeNet with extra convolutional unit")
    lenet_model = LeNetFCBN3(input_shape=input_shape, output_size=output_size)

    return lenet_model


def model_training(data, output_size, batch_size, optimizer, epochs, lr, verbose):
    """This function executes all the steps: model creation, compilation, training, and evaluation"""
    x_train, x_val, x_test, y_train, y_val, y_test = data
    input_shape = x_train.shape[1:]
    lenet_model = model_selection(input_shape, output_size)
    lenet_model.model_compilation(optimizer=optimizer, learning_rate=lr)
    history = lenet_model.train(x_train, y_train, x_val, y_val,
                                batch_size=batch_size,
                                epochs=epochs,
                                verbose=verbose)

    return lenet_model, history

