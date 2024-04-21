import matplotlib.pyplot as plt

class Metrics:
    def __init__(self, x_label, y_label_list):
        self.x_lable = x_label
        self.x_data = None
        self.y_label_list = y_label_list
        self.y_data = None

    def add(self, x_value, y_value_list):
        if len(y_value_list) != len(self.y_label_list):
            raise RuntimeError("metric length mismatch, require %d, provided %d"
                               % (len(self.y_label_list), len(y_value_list)))

        if not self.x_data and not self.y_data:
            self.x_data = []
            self.y_data = [[] for _ in self.y_label_list]

        self.x_data.append(x_value)
        for i in range(len(self.y_label_list)):
            self.y_data[i].append(y_value_list[i])

    def reset(self):
        self.x_data = []
        for _ in range(len(self.y_label_list)):
            self.y_data.append([])


    def plot(self):
        fig, ax = plt.subplots()

        ax.set_xlabel(self.x_lable)
        for y, legend in zip(self.y_data, self.y_label_list):
            ax.plot(self.x_data, y, marker='.', label=legend)
        ax.legend()

        plt.grid()
        plt.show()
