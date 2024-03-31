import tensorflow as tf


class Optimizer:
    def __init__(self, lr):
        self.lr = lr

    def apply_gradients(self, grads_and_vars):
        self.sgd(grads_and_vars)

    def sgd(self, grads_and_vars):
        for grad, param in grads_and_vars:
            param.assign_sub(self.lr * grad)
