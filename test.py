import numpy as np
import matplotlib.pyplot as plt
from Environment.Environment import Environment
from Model.PSO.Pso import Pso
from scipy.interpolate import griddata
from matplotlib import cm
from Environment.Parse import BuildingParser
import matplotlib.path as mplPath


def pso_test():
    np.random.seed(10)
    fig = plt.figure(figsize=(24, 10))
    ax1 = fig.add_subplot(141)
    ax2 = fig.add_subplot(142)
    ax3 = fig.add_subplot(143)
    ax4 = fig.add_subplot(144)
    envi = MyTest((100, 100))
    n = 5000
    model = Pso(n, 2, 0.5, 1000, 5, envi)
    # ax.scatter(model.pos[0], model.pos[1])
    ax1.scatter(model.pos[0], model.pos[1])
    ax1.set_title("Iter: 0 times")
    model.outputs(100)
    ax2.scatter(model.pos[0], model.pos[1])
    ax2.set_title("Iter: 100 times")
    model.outputs(100)
    ax3.scatter(model.pos[0], model.pos[1])
    ax3.set_title("Iter: 200 times")
    model.outputs(500)
    ax4.scatter(model.pos[0], model.pos[1])
    ax4.set_title("Iter: 300 times")
    print(f"{model.pos[:, model.global_best_index]}")
    print(model.global_best_fitness)
    # Plot some data on the axes.
    plt.savefig("PSO.png")
    plt.show()


def envi_test():
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    envi = MyTest((100, 100), 1000)
    points, z = envi.create((50, 50), 100)
    x, y = np.mgrid[0:100:1000j, 0:100:1000j]
    z = griddata(points, z, (x, y), method='cubic')
    print(z.shape)
    ax.plot_surface(x, y, z, cmap=cm.coolwarm)
    plt.show()

def pso_fitness_test():
    test = MyTest((100, 100))
    x, y, points, z = test.create((50, 50), 100)
    x, y, points, z = test.create((25, 25), 100)
    pos = np.random.rand(2, 1000) * 100
    fit = test.from_pos_getfit(pos)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(x, y, z)
    ax.scatter(pos[0], pos[1], fit, color='red', s=1)
    plt.show()

def pso_test():
    class environment(Environment):
        def __init__(self, size: tuple = (100, 100)):
            super(environment, self).__init__(size)
        def get_fitness(self, pso): # 适应度函数
            pos = pso.pos
            x, y, z = pos[0], pos[1], pos[2]
            A = 10
            return 3 * A + (x**2 - A * np.cos(2 * np.pi * x)) + (y**2 - A * np.cos(2 * np.pi * y)) + (z**2 - A * np.cos(2 * np.pi * z))
    envi = environment()
    model = Pso(100, 3, 0.01, 0.2, 0.1, envi)
    print(model.pos.shape)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    
    model.outputs(1000000)
    print(model.global_best_pos)
    print(model.global_best_fitness)
    ax.scatter(model.pos[0], model.pos[1], model.pos[2])
    plt.show()

def parser_test():
    parser = BuildingParser("/home/robinson/projects/QuadrotorMotion/File/building.json")
    envi = MyTest((100, 100))
    envi.create((50, 50), 100)
    envi.create_building(parser)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(envi.x, envi.y, envi.z)
    plt.show()

    
def pso_run():
    envi = MyTest((100, 100), 100j, (100., 70., 10), 10)
    envi.create((50, 50), 100)
    envi.point_lst.append(np.array([0., 0., 10.]).reshape(3, 1)) 
    epoch = 0
    while not envi.end():
        print(f"iter: {epoch}, pos: {envi.point_lst[-1].reshape(-1)}")
        epoch += 1
        pso = Pso(1000, 3, 0.5, 100, 0, 5, envi)
        point = pso.outputs(500)
        envi.point_lst.append(point)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(envi.x, envi.y, envi.z)
    points = np.array(envi.point_lst)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='red')
    plt.show()
    
def gridtest():
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    x, y = np.meshgrid(x, y)
    z = np.sin(x * np.pi) * np.cos(y * np.pi)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    x1 = [0.1, 0.2, 0.3, 0.4, 0.5]
    y1 = [0.1, 0.2, 0.3, 0.4, 0.5]
    fit = griddata((x.ravel(), y.ravel()), z.ravel(), (x1, y1), method='linear')
    print(fit)
    ax.scatter(x1, y1, fit+1, color='black')
    ax.plot_surface(x, y, z)
    plt.show()
if __name__ == "__main__":
    gridtest()
