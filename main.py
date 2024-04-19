import numpy as np
import matplotlib.pyplot as plt
from Environment.Environment import Environment
from Model.PSO.Pso import Pso
from matplotlib import cm
from scipy.interpolate import griddata


class MyTest(Environment):

    def __init__(self, size: tuple, number=100):
        super(MyTest, self).__init__(size)
        self.size = size

    def z(self, point: tuple = None, number=1000):
        if point is None:
            point = (50, 50)
        px, py = point
        points = np.random.rand(number, 2) * self.size[0]
        hi = 30 + 40 * np.random.rand(number)
        return points, hi * np.exp(-((points[:, 0] - px) / 20)**2 -
                                   ((points[:, 1] - py) / 20)**2)

    def get_fitness(self, pos):
        # return (1 - pos[0])**2 + 100 * (pos[1] - pos[0]**2)**2
        return self.z(pos[0], pos[1])


if __name__ == "__main__":
    # main()
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    envi = MyTest((100, 100), 1000)
    points, z = envi.z((50, 50), 100)
    x, y = np.mgrid[0:100:1000j, 0:100:1000j]
    z = griddata(points, z, (x, y), method='cubic')
    print(z.shape)
    ax.plot_surface(x, y, z, cmap=cm.coolwarm)
    plt.show()
