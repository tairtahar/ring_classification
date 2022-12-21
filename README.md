![tf](https://img.shields.io/badge/tensorflow--gpu-2.3.0-yellowgreen) ![python](https://img.shields.io/badge/Python-3.8-orange) 

# Ring Classification - valid or invalid appearance
This repository contains a CNN implementation for quality detection of ring images. 




## Usage

This repository contains two main scripts: `train.py` and `test.py`.

### `train.py`

This script serves to train our model, log the process' metrics and export a checkpoint. Eventually a test on unseen data is performed for evaluation.

### `test.py`
A script for evaluation an existing model on a dataset. It is possible to test the performance of the model.

To avoid leakage from training data, please indicate a new unseen data in the following way:
```
$ python test.py --data_path <your new data path>
```

For loading the model (.pb file):
```
$ python test.py --model_path <trained model .pb file>
```

The training process shown good performance as in the following image:

![results](https://github.com/tairtahar/ring_classification/blob/main/checkpoint5/results/Figure_2.png) 
