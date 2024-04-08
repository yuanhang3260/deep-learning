# coding=utf-8
import tensorflow as tf
import mnist_base
import datasets as ds
import mnist_images
import models


class MnistMlpModel(models.Model):
    def __init__(self, input_dim, output_dim, hidden_dim):
        self.w1 = tf.Variable(
            tf.random.normal(shape=(input_dim, hidden_dim), mean=0, stddev=0.01)
        )
        self.b1 = tf.Variable(tf.zeros(hidden_dim))
        self.w2 = tf.Variable(
            tf.random.normal(shape=(hidden_dim, output_dim), mean=0, stddev=0.01)
        )
        self.b2 = tf.Variable(tf.zeros(output_dim))

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.w1.shape[0]))
        h = models.relu(tf.matmul(x, self.w1) + self.b1)
        # Note softmax is moved to cross entropy loss.
        return tf.matmul(h, self.w2) + self.b2

    @property
    def trainable_variables(self):
        return [self.w1, self.b1, self.w2, self.b2]

    @property
    def losses(self):
        return []


def main():
    # Prepare dataset.
    mnist_train, mnist_test = mnist_images.load_images()
    train_images, train_labels = mnist_base.preprocess_data(*mnist_train)
    test_images, test_labels = mnist_base.preprocess_data(*mnist_test)
    train_dataset = ds.load_dataset((train_images, train_labels),
                                    batch_size=100,
                                    is_train=True)

    # Define model.
    #model = MnistMlpModel(input_dim=28 * 28, output_dim=10, hidden_dim=256)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, kernel_regularizer=tf.keras.regularizers.l2(0)))

    # Define loss function.
    #loss = lambda y, y_hat: tf.losses.sparse_categorical_crossentropy(y, y_hat, from_logits=True)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Define optimizer.
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

    # Start training.
    epochs, losses, train_accs, test_accs = [], [], [], []
    for epoch in range(50):
        loss_mean, train_acc = mnist_base.train_epoch(train_dataset, model, loss, optimizer)
        test_acc = mnist_base.accuracy(model(test_images), test_labels)
        print("epoch %d, training loss %f, train acc %f, test acc %f" %
              (epoch, loss_mean, train_acc, test_acc))

        epochs.append(epoch)
        losses.append(loss_mean)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

    mnist_base.plot_metric(epochs, losses, train_accs, test_accs)


if __name__ == "__main__":
    main()

