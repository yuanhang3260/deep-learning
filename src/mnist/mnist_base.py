# coding=utf-8
import tensorflow as tf

import base
import datasets as ds
import models
import training

from mnist import mnist_images


class MnistSimpleModel(models.Model):
    def __init__(self, input_dim, output_dim):
        self.w = tf.Variable(
            tf.random.normal(shape=(input_dim, output_dim), mean=0, stddev=0.01)
        )
        self.b = tf.Variable(tf.zeros(output_dim))

    def __call__(self, x):
        return models.softmax(
            tf.matmul(tf.reshape(x, (-1, self.w.shape[0])), self.w) + self.b)

    @property
    def trainable_variables(self):
        return [self.w, self.b]

    @property
    def losses(self):
        return []


# convert image pixel data from unit8 to float(0~1), and label to int32.
def preprocess_data(x, y):
    return tf.cast(x / 255.0, dtype='float32'), tf.cast(y, dtype='int32')


def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = tf.argmax(y_hat, axis=1)
    cmp = tf.cast(y_hat, y.dtype) == y
    return tf.reduce_sum(tf.cast(cmp, y.dtype)) / y.shape[0]


def train_epoch(dataset, model, loss, optimizer):
    metric = base.Accumulator(size=3)
    for x, y in dataset:
        with tf.GradientTape() as tape:
            y_hat = model(x)
            train_loss = tf.reduce_mean(loss(y, y_hat)) + tf.reduce_sum(model.losses)

        params = model.trainable_variables
        grads = tape.gradient(train_loss, params)
        optimizer.apply_gradients(zip(grads, params))

        # print("training loss %f, acc: %f" % (float(loss_mean), accuracy(y_hat, y)))
        metric.add(train_loss, accuracy(y_hat, y), 1)

    return metric[0] / metric[2], metric[1] / metric[2]


def main():
    # Prepare dataset.
    mnist_train, mnist_test = mnist_images.load_images()
    train_images, train_labels = preprocess_data(*mnist_train)
    test_images, test_labels = preprocess_data(*mnist_test)
    train_dataset = ds.load_dataset((train_images, train_labels),
                                    batch_size=128,
                                    is_train=True)

    # Define model.
    model = MnistSimpleModel(input_dim=28 * 28, output_dim=10)

    # Define loss function.
    loss = models.cross_entropy

    # Define optimizer.
    optimizer = training.Optimizer(learning_rate=0.1)

    # Start training.
    epochs, losses, train_accs, test_accs = [], [], [], []
    for epoch in range(15):
        loss_mean, train_acc = train_epoch(train_dataset, model, loss, optimizer)
        test_acc = accuracy(model(test_images), test_labels)
        print("epoch %d, training loss %f, train acc %f, test acc %f" %
              (epoch, loss_mean, train_acc, test_acc))

        epochs.append(epoch)
        losses.append(loss_mean)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

    base.plot_metrics(epochs, metrics=[losses, train_accs, test_accs],
                      legends=['loss', 'train acc', 'test acc'])


if __name__ == "__main__":
    main()
