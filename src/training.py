import tensorflow as tf


class Optimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def apply_gradients(self, grads_and_vars):
        self.sgd(grads_and_vars)

    def sgd(self, grads_and_vars):
        for grad, param in grads_and_vars:
            param.assign_sub(self.learning_rate * grad)
