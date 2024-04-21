# coding=utf-8
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import layers

from metrics import Metrics
import datasets as ds

from mnist import mnist_base, mnist_images


def main():
    # Prepare dataset.
    mnist_train, mnist_test = mnist_images.load_images()
    train_images, train_labels = mnist_base.preprocess_data(*mnist_train)
    test_images, test_labels = mnist_base.preprocess_data(*mnist_test)
    train_dataset = ds.load_dataset((train_images, train_labels),
                                    batch_size=128,
                                    is_train=True)

    # Define model.
    input_shape = train_images.shape[1:] + (1,)
    model = tf.keras.models.Sequential()
    model.add(keras.Input(shape=input_shape))
    model.add(tf.keras.layers.Conv2D(filters=6, kernel_size=3, activation='relu', padding='same'))
    model.add(tf.keras.layers.AvgPool2D(pool_size=2))
    model.add(tf.keras.layers.Conv2D(filters=12, kernel_size=3, activation='relu', padding='same'))
    model.add(tf.keras.layers.AvgPool2D(pool_size=2))
    model.add(tf.keras.layers.Flatten())
    #model.add(tf.keras.layers.Dense(120, activation='sigmoid'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10))

    model.summary()

    # Define loss function.
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Define optimizer.
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    #optimizer = tf.keras.optimizers.Adam()

    # Start training.
    metrics = Metrics(x_label='epoch', y_label_list=['loss', 'train acc', 'test acc'])
    for epoch in range(5):
        loss_mean, train_acc = mnist_base.train_epoch(train_dataset, model, loss, optimizer)
        test_acc = mnist_base.accuracy(model(test_images), test_labels)
        print("epoch %d, train loss %f, train acc %f, test acc %f" %
              (epoch, loss_mean, train_acc, test_acc))

        metrics.add(epoch, y_value_list=[loss_mean, train_acc, test_acc])

    metrics.plot()


def main2():
    # Model / data parameters
    num_classes = 10
    input_shape = (28, 28, 1)

    # Load the data and split it between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.summary()

    batch_size = 128
    epochs = 10

    model.compile(loss="categorical_crossentropy", optimizer="adam",
                  metrics=["accuracy"])

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
              validation_split=0.1)

    score = model.evaluate(x_test, y_test, verbose=0)
    print("test loss:", score[0])
    print("test accuracy:", score[1])


if __name__ == "__main__":
    main()
