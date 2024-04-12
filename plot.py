import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


def plot_3d_surface():
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make data.
    x = np.arange(-5, 5, 0.25)
    y = np.arange(-5, 5, 0.25)
    x, y = np.meshgrid(x, y)
    #z = np.sqrt(x ** 2 + y ** 2)
    z = x ** 2 + 2 * y ** 2

    # Plot the surface.
    ax.grid(True, which='both')
    ax.contour3D(x, y, z, zdir='z', offset=-5, cmap=cm.coolwarm)

    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=True)

    #ax.zaxis.set_major_formatter('{x:.02f}')

    plt.show()


def plot_2d():
    x = np.arange(0.0, 1.0, 0.01)
    y = x ** 0.7 + (1 - x) ** 0.3
    #y = 0.7 * np.log(x) + 0.3 * np.log(1 - x)

    fig, ax = plt.subplots()
    ax.plot(x, y)

    ax.set(xlabel='p', ylabel='cross entropy')
    ax.grid()

    plt.show()


def main():
    plot_2d()


if __name__ == "__main__":
    main()
