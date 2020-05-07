#  Copyright (c) 2020. Tuan-Hiep TRAN
import tensorflow as tf
import tensorflow_datasets as tfds

from A8.my_cnn import myCNN
from A8.prepare_dataset import load_mnist
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# TODO check how to convert numpy to correct form of tensorFlow data set

x_train, y_train = load_mnist("../csv/mnist/", "train")
x_test, y_test = load_mnist("../csv/mnist/", "t10k")

# Preprocess the data (these are Numpy arrays)
# x_train = x_train.reshape(60000, 784).astype('float32')
# x_test = x_test.reshape(10000, 784).astype('float32')

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

# Reserve 10,000 samples for validation
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]
# create the training datasets
dx_train = tf.data.Dataset.from_tensor_slices(x_train)
# apply a one-hot transformation to each label for use in the neural network
dy_train = tf.data.Dataset.from_tensor_slices(y_train)
# zip the x and y training data together and shuffle, batch etc.
train_dataset = tf.data.Dataset.zip((dx_train, dy_train)).shuffle(500).repeat().batch(30)
dataset_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
dataset_valid = tf.data.Dataset.from_tensor_slices((x_val, y_val))
# (x_train_keras, y_train_keras), (x_test_keras, y_test_keras) = tf.keras.datasets.mnist.load_data()
## Step 1: loading and preprocessing MNIST dataset
mnist_bldr = tfds.builder('mnist')
mnist_bldr.download_and_prepare()
datasets = mnist_bldr.as_dataset(shuffle_files=False)
print(datasets.keys())
mnist_train_orig, mnist_test_orig = datasets['train'], datasets['test']

mnist_train = mnist_train_orig.map(
    lambda item: (tf.cast(item['image'], tf.float32) / 255.0,  # RGB value range [0,255]
                  tf.cast(item['label'], tf.int32)))
mnist_test = mnist_test_orig.map(
    lambda item: (tf.cast(item['image'], tf.float32) / 255.0,
                  tf.cast(item['label'], tf.int32)))

# config the model
BUFFER_SIZE = 10000
BATCH_SIZE = 100
NUM_EPOCHS = 10
# tf.random.set_seed(1)
# mnist_train = mnist_train.shuffle(buffer_size=BUFFER_SIZE,
#                                   reshuffle_each_iteration=False)
# mnist_valid = mnist_train.take(10000).batch(BATCH_SIZE)
# mnist_train = mnist_train.skip(10000).batch(BATCH_SIZE)
# BATCH_SIZE = 64
# SHUFFLE_BUFFER_SIZE = 100
#
# dataset_train = dataset_train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
# dataset_valid = dataset_valid.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
# dataset_test = dataset_test.batch(BATCH_SIZE)

model = myCNN()
# model.fit(mnist_train, mnist_valid, NUM_EPOCHS)
# model.fit(dataset_train, dataset_valid, NUM_EPOCHS)
model.fit_np(x_train, y_train, x_val, y_val, NUM_EPOCHS)
# model.evaluate(mnist_test)
model.evaluate(dataset_test)
