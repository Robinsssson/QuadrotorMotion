import numpy as np
import matplotlib.pyplot as plt
from Environment.Environment import Environment
from Model.PSO.Pso import Pso
from matplotlib import cm
from scipy.interpolate import griddata
from Environment.Parse import BuildingParser
import matplotlib.path as mplPath
        
class MyTest(Environment):
    def __init__(self, size: tuple, number=1000j, final_pos: tuple[float, float]=(100., 100.), step: float=10):
        super(MyTest, self).__init__(size)
        self.size = size
        self.z = None
        self.mz = None
        self.bz = None
        self.x, self.y = np.mgrid[0:size[0]:number, 0:size[1]:number]
        self.final_pos = np.array(final_pos).reshape(2, 1)
        self.point_lst = []
        self.step = step
        self.builds = None

    def create(self, point: tuple = None, number: int =100):
        if point is None:
            point = (50, 50)
        px, py = point
        points = np.random.rand(number, 2) * self.size[0]
        # hi = 30 + 10 * np.random.uniform(0, 1, number)
        mz = np.exp(-((points[:, 0] - px) / 20)**2 - ((points[:, 1] - py) / 20)**2) * 20
        # self.height = Height(px, py)
        mz = griddata(points, mz, (self.x, self.y), method='linear')
        if self.mz is not None:
            mz = np.maximum(self.mz, mz)
        self.mz = mz
        self.z = mz.copy()
        final_pos_z = self.from_pos_getfit(self.final_pos)
        self.final_pos = np.array([self.final_pos[0], self.final_pos[1], final_pos_z]).reshape(3, 1)
        print(f'final_pos: {self.final_pos.reshape(-1)}')
    
    def create_building(self, paser: BuildingParser):
        bz = np.zeros_like(self.z)
        for build in paser.builds:
            trans_matrix = np.array([[np.cos(build.angle*np.pi/180), -np.sin(build.angle*np.pi/180)], [np.sin(build.angle*np.pi/180), np.cos(build.angle*np.pi/180)]])
            pos = np.array(build.pos)
            h = self.from_pos_getfit(pos)
            build.local_h = h
            x1 = np.array([-build.size[0] / 2, -build.size[1] / 2])         # [60, 40] [10, 10] # [55, 35]
            x2 = np.array([build.size[0] / 2, -build.size[1] / 2])          # [65, 35]
            x3 = np.array([build.size[0] / 2, build.size[1] / 2])           # [65, 45]
            x4 = np.array([-build.size[0] / 2, build.size[1] / 2])          # [55, 45]
            x1 = trans_matrix @ x1.reshape(2, 1) + pos.reshape(2, 1)        # [[0, -1], [1, 0]] @ [55, 35]
            x2 = trans_matrix @ x2.reshape(2, 1) + pos.reshape(2, 1) 
            x3 = trans_matrix @ x3.reshape(2, 1) + pos.reshape(2, 1) 
            x4 = trans_matrix @ x4.reshape(2, 1) + pos.reshape(2, 1)        # 如果x，y在x1, x2, x3, x4的凸包内，则z = h + build.height ，否则 z = 0， 生成一个矩阵z
            path = mplPath.Path(np.vstack([x1.T, x2.T, x3.T, x4.T]))
            mask = path.contains_points(np.array([self.x.ravel(), self.y.ravel()]).T) # 生成一个mask from x, y 
            bz.ravel()[mask] = h + build.height
        z = np.maximum(self.mz, bz)
        self.z = z
        self.bz = bz
        final_pos_z = self.from_pos_getfit(self.final_pos)
        self.final_pos = np.array([self.final_pos[0], self.final_pos[1], final_pos_z]).reshape(3, 1)
        print(f'final_pos: {self.final_pos.reshape(-1)}')
        self.builds = paser.builds
        
    def _f_angle1(self, k: float, ang: float): # k: 惩罚系数， ang: 角度, fit值域为[0, k]
        if len(self.point_lst) < 2:
            return 0
        p1, p2 = self.point_lst[-2][:-1], self.point_lst[-1][:-1]
        pos = self.pos[:-1]
        px1 = pos - p2
        px2 = p2 - p1
        # Calculate the cosine of the angle between px1 and px2
        cos = np.sum(px1 * px2, axis=0) / (np.linalg.norm(px1, axis=0) * np.linalg.norm(px2, axis=0))
        # Determine which elements of cos are less than the cosine of the given angle
        mask = cos < np.cos(ang) # 限制角度 角度小于ang
        # Create the fit array and set the elements where mask is True to k
        fit = np.zeros_like(pos[0])
        fit[mask] = k
        return fit
    
    def _f_angle2(self, k, ang): # k: 惩罚系数， ang: 角度， fit值域为[0, k]
        p1 = self.point_lst[-1]
        pos = self.pos - p1
        pos_xy, pos_z = pos[:-1], pos[-1]
        dxy = np.linalg.norm(pos_xy, axis=0)
        theta = np.arctan2(pos_z, dxy)
        assert np.size(theta) == np.size(dxy)
        mask = np.abs(theta) > ang  # 限制角度 角度小于ang
        fit = np.zeros_like(pos[0])
        fit[mask] = k
        return fit
         
    def _f_step(self, l): # l: 步长， fit值域为[0, |d-l|]
        pos = self.pos
        p1 = self.point_lst[-1]
        pos = pos - p1
        d = np.linalg.norm(pos, axis=0)
        fit = np.abs(d-l)
        return fit
    
    def _f_height(self, k: float, min_h: float): # k: 惩罚系数， min_h: 最小高度 fit值域为[h-min_h， inf]
        pos = self.pos
        h = self.from_pos_getfit(pos)
        pos_z = pos[-1]
        hk = pos_z - h
        
        fit = np.zeros_like(pos[0])
        fit[hk < 0] = np.inf
        fit[np.logical_or(hk < min_h, hk > 0)] = k
        fit[hk >= min_h] = hk[hk >= min_h] - min_h
        return fit
    
    def _f_close(self, final_pos: np.array): # final_pos: 最终位置 fit值域为[0, 1]
        p1 = self.point_lst[-1]
        pos = self.pos
        l = np.linalg.norm(final_pos.reshape(3, 1) - p1, axis=0)
        lx = np.linalg.norm(pos - final_pos.reshape(3, 1), axis=0)
        fit = lx/l
        return fit
    
    def _serach_building(self):
        fit = np.zeros_like(self.pos[0])
        pos = self.pos
        for build in self.builds:
            fit += build.weight
            if build.serached:
                fit -= build.weight
                continue
            bp = np.array((build.pos[0], build.pos[1], build.local_h)).reshape(3, 1)
            pls = self.point_lst[-1]
            dis = np.linalg.norm(pls - bp, axis=0) < build.distance
            if np.any(dis):
                build.serached = True
                print(f'find building: {build.label}, {dis}')
                fit -= build.weight
                continue
            dis = np.linalg.norm(pos - bp, axis=0) < build.distance
            fit[dis] -= build.weight
        return fit
             
    def _close_building(self):
        fit = np.zeros_like(self.pos[0])
        pos = self.pos
        lp = self.point_lst[-1]
        for build in self.builds:
            if build.serached:
                continue
            bp = np.array((build.pos[0], build.pos[1], build.local_h)).reshape(3, 1)
            l = np.linalg.norm(bp - lp, axis=0)
            lx = np.linalg.norm(pos - bp, axis=0)
            fit += lx/l
        return fit     
    
    def get_fitness(self, pso: Pso):
        pos = pso.pos
        self.pos = pos
        fit = self._f_angle1(10, np.pi/4) + self._f_angle2(10, np.pi/4) + 10 * self._f_step(self.step) + self._f_height(10, 3) + 10 * self._f_close(self.final_pos) +20 * self._serach_building() + 20 * self._close_building()
        return fit
    
    def from_pos_getfit(self, pos):
        x, y = pos[0], pos[1] # [1, D]
        fit = griddata((self.x.ravel(), self.y.ravel()), self.z.ravel(), (x, y), method='linear')
        # fit = self.height(x, y)
        return fit
    
    def end(self):
        return np.linalg.norm(self.point_lst[-1] - self.final_pos.reshape(3, 1), axis=0) < self.step
    
    def create_point_lst(self, point: np.array):
        assert point.shape == (2, 1)
        pz = self.from_pos_getfit(point) + 2
        print(f'pz: {pz}')
        point = np.array([point[0], point[1], pz]).reshape(3, 1)
        self.point_lst.append(point)

def main():
    envi = MyTest((100, 100), 200j, (80.1, 80.1), 5)
    parser = BuildingParser("/home/robinson/projects/QuadrotorMotion/File/building.json")
    envi.create((50, 50), 200)
    envi.create_building(parser)
    p = np.array([10., 10.]).reshape(2, 1)
    envi.create_point_lst(p)
    epoch = 0
    while not envi.end():
    # for _ in range(10):
        centor_point = envi.point_lst[-1].reshape(-1)
        print(f"iter: {epoch}, pos: {centor_point}")
        epoch += 1
        pso = Pso(200, 3, 0.5, centor_point, 10, 5, envi)
        point = pso.outputs(300)
        envi.point_lst.append(point)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(envi.x, envi.y, envi.z, cmap=plt.cm.coolwarm, linewidth=0.1, zorder=1)
    points = np.array(envi.point_lst)
    ax.plot(points[:, 0], points[:, 1], points[:, 2], 'b.', markersize=10, label='path', zorder=10)
    plt.show()
    for build in envi.builds:
        print(build)
    
if __name__ == "__main__":
    main()
