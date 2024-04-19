import numpy as np
import matplotlib.pyplot as plt
from Environment.Environment import Environment
from Model.PSO.Pso import Pso
from scipy.interpolate import griddata
from matplotlib import cm
from Environment.Parse import BuildingParser
import matplotlib.path as mplPath

class MyTest(Environment):

    def __init__(self, size: tuple, number=1000j, final_pos=(100, 100, 50), step=10):
        super(MyTest, self).__init__(size)
        self.size = size
        self.z = None
        self.x, self.y = np.mgrid[0:size[0]:number, 0:size[1]:number]
        self.final_pos = np.array(final_pos).reshape(3, 1)
        self.point_lst = []
        self.step = step
    def create(self, point: tuple = None, number=100):
        if point is None:
            point = (50, 50)
        px, py = point
        points = np.random.rand(number, 2) * self.size[0]
        hi = 30 + 40 * np.random.rand(number)
        z = hi * np.exp(-((points[:, 0] - px) / 20)**2 - ((points[:, 1] - py) / 20)**2)
        z = griddata(points, z, (self.x, self.y), method='cubic')
        if self.z is not None:
            z = np.maximum(self.z, z)
        self.z = z
    
    def create_building(self, paser: BuildingParser):
        z = np.zeros_like(self.x)
        for build in paser.builds:
            trans_matrix = np.array([[np.cos(build.angle*np.pi/180), -np.sin(build.angle*np.pi/180)], [np.sin(build.angle*np.pi/180), np.cos(build.angle*np.pi/180)]])
            pos = np.array(build.pos)
            h = self.from_pos_getfit(pos)
            x1 = np.array([-build.size[0] / 2, -build.size[1] / 2]) # [60, 40] [10, 10] # [55, 35]
            x2 = np.array([build.size[0] / 2, -build.size[1] / 2]) # [65, 35]
            x3 = np.array([build.size[0] / 2, build.size[1] / 2])   # [65, 45]
            x4 = np.array([-build.size[0] / 2, build.size[1] / 2]) # [55, 45]
            x1 = trans_matrix @ x1.reshape(2, 1) + pos.reshape(2, 1)        # [[0, -1], [1, 0]] @ [55, 35]
            x2 = trans_matrix @ x2.reshape(2, 1) + pos.reshape(2, 1) 
            x3 = trans_matrix @ x3.reshape(2, 1) + pos.reshape(2, 1) 
            x4 = trans_matrix @ x4.reshape(2, 1) + pos.reshape(2, 1) 

            # 如果x，y在x1, x2, x3, x4的凸包内，则z = h + build.height ，否则 z = 0， 生成一个矩阵z
            path = mplPath.Path(np.vstack([x1.T, x2.T, x3.T, x4.T]))
            mask = path.contains_points(np.array([self.x.ravel(), self.y.ravel()]).T) # 生成一个mask from x, y 
            z.ravel()[mask] = h + build.height
        z = np.maximum(self.z, z)
        self.z = z
        
    def _f_angle1(self, k, ang):
        if np.size(self.point_lst) < 2 * 3:
            return 0
        p1, p2 = self.point_lst[-2][:-1], self.point_lst[-1][:-1]
        pos = self.pos[:-1]
        px1 = pos - p2
        px2 = p2 - p1
        # Calculate the cosine of the angle between px1 and px2
        cos = np.sum(px1 * px2, axis=0) / (np.linalg.norm(px1, axis=0) * np.linalg.norm(px2, axis=0))
        # Determine which elements of cos are less than the cosine of the given angle
        mask = cos < np.cos(ang)
        # Create the fit array and set the elements where mask is True to k
        fit = np.zeros_like(pos[0])
        fit[mask] = k
        return fit
    
    def _f_angle2(self, k, ang):
        if np.size(self.point_lst) < 1 * 3:
            return 0
        p1 = self.point_lst[-1]
        pos = self.pos - p1
        pos_xy, pos_z = pos[:-1], pos[-1]
        dxy = np.linalg.norm(pos_xy, axis=0)
        theta = np.arctan2(pos_z, dxy)
        assert np.size(theta) == np.size(dxy)
        mask = np.abs(theta) < ang
        fit = np.zeros_like(pos[0])
        fit[mask] = k
        return fit
         
    def _f_step(self, l):
        if np.size(self.point_lst) < 1 * 3:
            return 0
        pos = self.pos
        p1 = self.point_lst[-1]
        pos = pos - p1
        d = np.linalg.norm(pos, axis=0)
        fit = np.abs(d-l)
        return fit
    
    def _f_height(self, k, min_h):
        pos = self.pos
        h = self.from_pos_getfit(pos)
        pos_z = pos[-1]
        hk = pos_z - h
        fit = np.zeros_like(pos[0])
        fit[hk < 0] = np.inf
        fit[np.logical_or(hk < min_h, hk > 0)] = k
        fit[hk >= min_h] = hk[hk >= min_h] - min_h
        return fit
    
    def f_close(self, final_pos):
        if np.size(self.point_lst) < 1 * 3:
            return np.ones_like(self.pos[0])
        p1 = self.point_lst[-1]
        pos = self.pos
        l = np.linalg.norm(final_pos.reshape(3, 1) - p1, axis=0)
        lx = np.linalg.norm(pos - p1, axis=0)
        fit = lx/l
        return fit
    
    def get_fitness(self, pso: Pso):
        pos = pso.pos
        self.pos = pos
        fit = 2 * self._f_angle1(10, np.pi/4) + 2 * self._f_angle2(10, np.pi/4) +3 * self._f_step(self.step) +4 * self._f_height(10, 10) + 5 * self.f_close(self.final_pos)
        return fit
    
    def from_pos_getfit(self, pos):
        x, y = pos[0], pos[1]
        fit = griddata((self.x.ravel(), self.y.ravel()), self.z.ravel(), (x, y), method='nearest')
        return fit
    
    def end(self):
        return np.linalg.norm(self.point_lst[-1] - self.final_pos.reshape(3, 1), axis=0) < self.step


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
    envi = MyTest((100, 100), 1000j, (100., 70., 10), 10)
    envi.create((50, 50), 100)
    envi.point_lst.append(np.array([0., 0., 10.]).reshape(3, 1)) 
    epoch = 0
    while not envi.end():
        print(f"iter: {epoch}, pos: {envi.point_lst[-1].reshape(-1)}")
        epoch += 1
        pso = Pso(10000, 3, 0.5, 100, 0, 5, envi)
        point = pso.outputs(5000)
        envi.point_lst.append(point)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(envi.x, envi.y, envi.z)
    points = np.array(envi.point_lst)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='red')
    plt.show()
    
if __name__ == "__main__":
    pso_run()
