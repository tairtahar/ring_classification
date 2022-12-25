import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf


def load_images_from_folder(path: str):
    fns = os.listdir(path)
    im_size = 32, 32
    img_list = []

    for fn in fns:
        img = cv2.imread(os.path.join(path, fn))
        img = cv2.resize(img, im_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

        img_list.append(img)
    return img_list


def load_data(path="data", train=True):
    path_valid = "Ring X - Good - No Bugs"
    path_invalid = "Ring X - Bad - With Bugs"
    valid_list = load_images_from_folder(os.path.join(path, path_valid))
    y_valid = np.ones(len(valid_list), dtype=int)
    valid_imgs = np.vstack(np.expand_dims(valid_list, 1))
    invalid_list = load_images_from_folder(os.path.join(path, path_invalid))
    invalid_imgs = np.vstack(np.expand_dims(invalid_list, 1))
    y_invalid = np.zeros(len(invalid_list), dtype=int)
    if train:
        print("Data set conatins " + str(len(valid_list)) + " valid images and " +
              str(len(invalid_list)) + " invalid images")

    return valid_imgs, y_valid, invalid_imgs, y_invalid


def data_prepare(path="data", train=True, seed=123):
    valid_imgs, y_valid, invalid_imgs, y_invalid = load_data(path, train)
    X_data = np.vstack((valid_imgs, invalid_imgs))
    y_data_1d = np.hstack((y_valid, y_invalid))
    np.random.RandomState(seed=seed)
    perm = np.random.permutation(range(len(y_data_1d)))
    X_data = X_data[perm, :, :]
    y_data_1d = y_data_1d[perm]

    # One hot encoding of classes:
    y_data = np.zeros((y_data_1d.size, y_data_1d.max() + 1))
    y_data[np.arange(y_data_1d.size), y_data_1d] = 1

    x_train, x_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.15)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.22)
    x_train = tf.expand_dims(x_train, axis=3, name=None)
    x_test = tf.expand_dims(x_test, axis=3, name=None)
    x_val = tf.expand_dims(x_val, axis=3, name=None)
    data = x_train, x_val, x_test, y_train, y_val, y_test
    return data


def test_data_load(path="data", train=True, seed=123):
    data = data_prepare(path, train, seed)
    x_train, x_val, x_test, y_train, y_val, y_test = data
    return x_test, y_test


def print_evaluation(model, x_test, y_test, verbose=0):
    """
    Evaluation of model performance
    :param x_test: input images for testing
    :param y_test: ground truth for testing
    :param verbose: verbosity for testing phase
    :return: NA
    """
    score = model.evaluate(x_test, y_test, verbose=verbose)
    print('test loss:', score[0])
    print('test accuracy:', score[1])
