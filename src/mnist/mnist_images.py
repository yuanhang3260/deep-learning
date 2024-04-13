# coding=utf-8
import math
import tensorflow as tf
import matplotlib.pyplot as plt

fashion_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                  'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
num_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def get_fashion_mnist_labels(category, labels):
    if category == 'num':
        return [num_labels[int(i)] for i in labels]
    elif category == 'fashion':
        return [fashion_labels[int(i)] for i in labels]


def show_images(category, images, labels, num):
    img = images[:num]
    titles = get_fashion_mnist_labels(category, labels[:num])

    fig = plt.figure(figsize=(16, 8))
    for i in range(0, num):
        ax = fig.add_subplot(math.ceil(num / 8.0), 8, i + 1)
        ax.set_title(titles[i])
        ax.imshow(img[i], cmap=plt.cm.gray)

    plt.show()


def load_images(category='num'):
    if category == 'num':
        return tf.keras.datasets.mnist.load_data()
    elif category == 'fashion':
        return tf.keras.datasets.fashion_mnist.load_data()


def main():
    category = 'num'
    mnist_train, mnist_test = load_images(category=category)
    print(mnist_train[0].shape)
    print(mnist_train[1].shape)

    show_images(category, images=mnist_train[0], labels=mnist_train[1], num=16)


if __name__ == "__main__":
    main()
