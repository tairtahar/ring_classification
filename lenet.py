import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras


class LeNet(tf.keras.Model):
    """Original LeNet network implementation"""

    def __init__(self, input_shape: int, output_size=10):
        """
        Initialization function for a LeNet instance
        :param input_shape: network inputs images shape
        :param output_size: number of possible classes for the classification
        """
        super(LeNet, self).__init__()
        if input_shape is None:
            input_shape = (32, 32)
        self.input1 = layers.Input(shape=input_shape)
        self.c1 = layers.Conv2D(filters=6,
                                input_shape=input_shape,
                                kernel_size=(5, 5),
                                padding='valid',
                                activation='sigmoid'
                                )
        self.s2 = layers.AveragePooling2D(padding='valid')
        self.c3 = layers.Conv2D(filters=16,
                                kernel_size=(3, 3),
                                padding='valid',
                                activation='sigmoid')
        self.s4 = layers.AveragePooling2D(padding='valid')
        self.flatten = layers.Flatten()
        self.c5 = layers.Dense(units=120,
                               activation='sigmoid')
        self.f6 = layers.Dense(
            units=84,
            activation='sigmoid')
        self.output_layer = layers.Dense(
            units=output_size,
            activation=tf.nn.softmax)

    def call(self, inputs, training=False):
        """
        When training/testing the network, this function is called.
        :param inputs: The network inputs images
        :param training: flag for training/testing time. When True then it is training. When False it is inference time.
        :return: This function returns the outputs of the network after processing
        """
        x = self.c1(inputs)
        x = self.s2(x)
        x = self.c3(x)
        x = self.s4(x)
        x = self.flatten(x)
        x = self.c5(x)
        x = self.f6(x)
        return self.output_layer(x)

    def model_compilation(self, optimizer: str, learning_rate):
        """
        model compilation. Define optimizer and metric.
        :param optimizer: can be any of keras optimizers such as adam/ adagrad etc.
        :return: NA
        """
        if optimizer == 'adam':
            self.compile(optimizer=keras.optimizers.Adam(lr=learning_rate),
                         loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                         metrics=['accuracy'])

        print("compilation done")

    def train(self, x_train, y_train, x_val, y_val, batch_size, epochs=5, verbose=0, min_lr=0.000001):
        """
        Define the callbacks and train the model
        :param x_train: input images for training
        :param y_train: classes for training
        :param x_val: input images for validation
        :param y_val: classes for validation
        :param batch_size: Batch size for training
        :param epochs: number of epochs for training
        :param verbose: verbosity of training process
        :return: history of training
        """
        callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=15, min_lr=min_lr),
                     tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, verbose=1)]
        history = self.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                           validation_data=(x_val, y_val),
                           verbose=verbose,
                           callbacks=callbacks
                           )
        print("model training is done")
        return history

    def print_evaluation(self, x_test, y_test, verbose=0):
        """
        Evaluation of model performance
        :param x_test: input images for testing
        :param y_test: ground truth for testing
        :param verbose: verbosity for testing phase
        :return: NA
        """
        score = self.evaluate(x_test, y_test, verbose=verbose)
        print('test loss:', score[0])
        print('test accuracy:', score[1])


class LeNetFCBN3(LeNet):
    """LeNet with batchnorm on 2 conv layers and 2 fully connected layers"""

    def __init__(self, input_shape: int, output_size=1):
        """
        Initialization function for a LeNet instance
        :param input_shape: network inputs images shape
        :param output_size: number of possible classes for the classification
        """
        super(LeNet, self).__init__()
        if input_shape is None:
            input_shape = (32, 32)
        self.input1 = layers.Input(shape=input_shape)
        self.c1 = layers.Conv2D(filters=32,
                                input_shape=input_shape,
                                kernel_size=(3, 3),
                                padding='valid',
                                activation='relu'
                                )
        self.affine1 = layers.BatchNormalization()
        self.s2 = layers.MaxPool2D(padding='valid')
        self.c3 = layers.Conv2D(filters=64,
                                kernel_size=(3, 3),
                                padding='valid',
                                activation='relu')
        self.s4 = layers.MaxPool2D(padding='valid')
        # self.affine2 = layers.BatchNormalization()
        self.affine2 = layers.SpatialDropout2D(0.2)
        self.c_add = layers.Conv2D(filters=128,
                                   input_shape=input_shape,
                                   kernel_size=(3, 3),
                                   padding='valid',
                                   activation='relu'
                                   )
        # self.affine_add = layers.BatchNormalization()
        self.affine_add = layers.SpatialDropout2D(0.2)
        self.s_add = layers.MaxPool2D(padding='valid')

        self.flatten = layers.Flatten()
        self.c5 = layers.Dense(units=240,
                               activation='relu')
        self.c5_drop = layers.Dropout(0.15)
        self.affine3 = layers.BatchNormalization()
        self.affine4 = layers.BatchNormalization()

        self.f6 = layers.Dense(
            units=84,
            activation='relu')
        self.output_layer = layers.Dense(
            units=output_size,
            activation=tf.nn.softmax)

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.affine1(x)
        x = tf.keras.activations.relu(x)
        x = self.s2(x)
        x = self.c3(x)
        x = self.affine2(x)
        x = self.c_add(x)
        x = self.affine_add(x)
        x = self.s_add(x)
        x = tf.keras.activations.relu(x)
        x = self.s4(x)
        x = self.flatten(x)
        x = self.c5(x)
        x = self.c5_drop(x)
        x = self.affine3(x)
        x = tf.keras.activations.relu(x)
        x = self.f6(x)
        x = self.affine4(x)
        x = tf.keras.activations.softmax(x)
        return self.output_layer(x)
