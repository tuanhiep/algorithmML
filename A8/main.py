#  Copyright (c) 2020. Tuan-Hiep TRAN
import argparse
import os
import time
from A8.my_cnn import myCNN
from A8.prepare_dataset import load_mnist

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

parser = argparse.ArgumentParser(description="Convolutional Neural Network")
parser.add_argument("-n_epoch", "--n_epoch", type=int, required=True, help="Number of epoches for training")
parser.add_argument("-batch_size", "--batch_size", type=int, required=True, help="Batch size for training")
parser.add_argument("-kernel_size", "--kernel_size", type=int, required=True, help="Kernel size for training")
parser.add_argument("-strides_conv1", "--strides_conv1", type=int, required=True,
                    help="Strides for convolutional layer 1")
parser.add_argument("-pool_1_size", "--pool_1_size", type=int, required=True, help="Size for pooling layer 1")
parser.add_argument("-strides_conv2", "--strides_conv2", type=int, required=True,
                    help="Strides for convolutional layer 2")
parser.add_argument("-pool_2_size", "--pool_2_size", type=int, required=True, help="Size for pooling layer 2")
args = parser.parse_args()

# load dataset
x_train, y_train = load_mnist("../csv/mnist/", "train")
x_test, y_test = load_mnist("../csv/mnist/", "t10k")

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

# Reserve 10,000 samples for validation
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

# config the model
BUFFER_SIZE = 10000
BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.n_epoch
kernel_size = (args.kernel_size, args.kernel_size)
strides_conv1 = (args.strides_conv1, args.strides_conv1)
pool_1_size = (args.pool_1_size, args.pool_1_size)
strides_conv2 = (args.strides_conv2, args.strides_conv2)
pool_2_size = (args.pool_2_size, args.pool_2_size)

model = myCNN(kernel_size, strides_conv1, pool_1_size, strides_conv2, pool_2_size)
# Start to measure running time of training process
start_time = time.time()
model.fit_np(x_train, y_train, x_val, y_val, BATCH_SIZE, NUM_EPOCHS)
print("Time for processing is  %s seconds" % (time.time() - start_time))
model.evaluate_np(x_test, y_test)
