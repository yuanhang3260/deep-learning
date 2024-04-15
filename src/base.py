import matplotlib.pyplot as plt

class Accumulator:
    def __init__(self, size):
        self.data = [0.0] * size

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def plot_metrics(epochs, metrics, legends):
    fig, ax = plt.subplots()

    ax.set_xlabel('epoch')
    for y, legend in zip(metrics, legends):
        ax.plot(epochs, y, marker='.', label=legend)
    ax.legend()

    plt.grid()
    plt.show()

