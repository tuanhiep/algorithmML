#  Copyright (c) 2020. Tuan-Hiep TRAN

import tensorflow as tf
import tensorflow_datasets as tfds

print(tf.__version__)

# loading and preprocessing MNIST dataset
mnist_bldr = tfds.builder('mnist')
mnist_bldr.download_and_prepare()
datasets = mnist_bldr.as_dataset(shuffle_files=False)
print(datasets.keys())
mnist_train_orig, mnist_test_orig = datasets['train'], datasets['test']
print(mnist_train_orig, mnist_test_orig)

mnist_train = mnist_train_orig.map(
    lambda item: (tf.cast(item['image'], tf.float32) / 255.0,  # RGB value range [0,255]
                  tf.cast(item['label'], tf.int32)))
mnist_test = mnist_test_orig.map(
    lambda item: (tf.cast(item['image'], tf.float32) / 255.0,
                  tf.cast(item['label'], tf.int32)))

BUFFER_SIZE = 10000
BATCH_SIZE = 64
NUM_EPOCHS = 2
tf.random.set_seed(1)
mnist_train = mnist_train.shuffle(buffer_size=BUFFER_SIZE, reshuffle_each_iteration=False)
mnist_valid = mnist_train.take(10000).batch(BATCH_SIZE)
mnist_train = mnist_train.skip(10000).batch(BATCH_SIZE)
print(mnist_train, mnist_valid, mnist_test)

# construct Convolutional Neural Network
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(
    filters=32, kernel_size=(5, 5),
    strides=(1, 1), padding='same',
    data_format='channels_last',
    name='conv_1', activation='relu'))
model.add(tf.keras.layers.MaxPool2D(
    pool_size=(2, 2), name='pool_1'))
model.add(tf.keras.layers.Conv2D(
    filters=64, kernel_size=(5, 5),
    strides=(1, 1), padding='same',
    name='conv_2', activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), name="pool_2"))
model.compute_output_shape(input_shape=(16, 28, 28, 1))
# add flatten layer before adding full connected layer
model.add(tf.keras.layers.Flatten())
model.compute_output_shape(input_shape=(16, 28, 28, 1))
model.add(tf.keras.layers.Dense(units=1024, name='fc_1', activation='relu'))
model.add(tf.keras.layers.Dropout(rate=0.5))
model.add(tf.keras.layers.Dense(units=10, name='fc_2', activation="softmax"))
tf.random.set_seed(1)
model.build(input_shape=(None, 28, 28, 1))
model.compute_output_shape(input_shape=(16, 28, 28, 1))
# compile model
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=["accuracy"])
history = model.fit(mnist_train, epochs=NUM_EPOCHS,
                    validation_data=mnist_valid,
                    shuffle=True)
# get the accuracy
test_results = model.evaluate(mnist_test.batch(20))
print('\nTest Acc. {:.2f}%'.format(test_results[1] * 100))
