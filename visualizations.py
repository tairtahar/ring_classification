import matplotlib.pyplot as plt
import numpy as np


def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


def plot_accuracy(history, network):
    """This function receives list of training accuracy histories, and a list of network, such that histories[i]
    contains the accuracy obtained by training network[i]. Hence, histories and networks are lists with the same
    length."""
    plt.plot(history)
    plt.yticks(np.arange(0.5, 1, 0.02))
    plt.grid(which='major', linestyle=':', alpha=0.6)
    plt.ylim([0.7, 1])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(network, loc='lower right')
    plt.show()


def plot_learning_curves(history):
    accuracy = list(history.history['accuracy'])
    val_accuracy = list(history.history['val_accuracy'])
    acc = plt.plot(accuracy)
    acc_val = plt.plot(val_accuracy)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid(which='major', linestyle=':', alpha=0.6)
    plt.ylim([0, 1])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train accuracy', 'validation accuracy'], loc='lower right')
    plt.show()
