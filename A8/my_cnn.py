#  Copyright (c) 2020. Tuan-Hiep TRAN
import tensorflow as tf


class myCNN:

    def __init__(self, kernel_size, strides_conv1, pool_1_size, strides_conv2, pool_2_size):
        self.model = tf.keras.Sequential()
        # first convolutional layer using kernel size 3×3, strides 1×1, valid padding mode, and output channel size 4
        self.model.add(tf.keras.layers.Conv2D(
            filters=4, kernel_size=kernel_size,
            strides=strides_conv1, padding='valid',
            data_format='channels_last',
            name='conv_1', activation='relu'))
        self.model.compute_output_shape(input_shape=(16, 28, 28, 1))
        # first pooling layer using max pooling with pooling size 2 × 2 and strides 2 × 2.
        self.model.add(tf.keras.layers.MaxPool2D(
            pool_size=pool_1_size, strides=pool_1_size, name='pool_1'))
        self.model.compute_output_shape(input_shape=(16, 28, 28, 1))
        # second convolutional layer using kernel size 3×3 , strides 3×3 , valid padding mode, and output channel size 2
        self.model.add(tf.keras.layers.Conv2D(
            filters=2, kernel_size=kernel_size,
            strides=strides_conv2, padding='valid',
            name='conv_2', activation='relu'))
        self.model.compute_output_shape(input_shape=(16, 28, 28, 1))
        # second pooling layer using max pooling with pooling size  4×4 , and strides  4×4
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=pool_2_size, strides=pool_2_size, name="pool_2"))
        self.model.compute_output_shape(input_shape=(16, 28, 28, 1))
        self.model.add(tf.keras.layers.Flatten())
        self.model.compute_output_shape(input_shape=(16, 28, 28, 1))
        # fully connected layer from the second pooling layer with output channel size 10
        self.model.add(tf.keras.layers.Dense(units=10, name='fc_1', activation="softmax"))
        # print(tf.__version__)
        tf.random.set_seed(1)
        self.model.build(input_shape=(None, 28, 28, 1))
        self.model.compute_output_shape(input_shape=(16, 28, 28, 1))
        self.model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                           metrics=["accuracy"])

    def fit(self, data_train, data_valid, NUM_EPOCHS):
        history = self.model.fit(data_train, epochs=NUM_EPOCHS,
                                 validation_data=data_valid,
                                 shuffle=True)

    def fit_np(self, x_train, y_train, x_val, y_val, BATCH_SIZE, NUM_EPOCHS):
        history = self.model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,
                                 validation_data=(x_val, y_val),
                                 shuffle=True)

    def evaluate(self, data_test):
        test_results = self.model.evaluate(data_test.batch(20))
        print('\nTest loss {}'.format(test_results[0]))
        print('\nTest Accuracy {:.2f}%'.format(test_results[1] * 100))
        return test_results

    def evaluate_np(self, x_test, y_test):
        test_results = self.model.evaluate(x_test, y_test, verbose=0)
        print('\nTest loss:', test_results[0])
        print('\nTest Accuracy {:.2f}%'.format(test_results[1] * 100))
        return test_results
