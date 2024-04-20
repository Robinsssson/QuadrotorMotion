import numpy as np
import Environment.Environment as Environment
from tqdm import tqdm

class Pso:
    def __init__(self, number: int, dimension: int, w: float, center, rag: float, max_vel: float, environment: Environment) -> None:
        self.number = number                                                            # 设置私有数目
        self.dimension = dimension                                                      # 设置变量维度
        self.environment = environment                                                  # 添加场地条件
        assert len(center) == dimension                                                  # 判断维度是否相等
        min_pos = np.array(center).reshape(dimension, 1) - rag                          # 设置最小位置
        max_pos = np.array(center).reshape(dimension, 1) + rag                          # 设置最大位置
        self.pos = np.random.uniform(0, 1, (dimension, number))                         # using max_pos. create pos[d, n]
        self.pos = self.pos * (max_pos - min_pos) + min_pos
        self.v = np.random.uniform(-max_vel, max_vel, (dimension, number))              # using max_vec. create v[d, n]
        self.max_vel = max_vel
        self.w = w
        self.c1 = self.c2 = 2
        self.fitness = environment.get_fitness(self)                                    # get fitness[n]
        self.private_best_fitness = self.fitness.copy()                                 # private_best_fitness[n] = function(pos)
        self.private_best_pos = self.pos.copy()                                         # default make private_best_pos[d, n] = pos[d, n]
        self.global_best_index = np.argmin(self.private_best_fitness)                   # self.global_best_index = index of min private_best_fitness
        self.global_best_fitness = self.private_best_fitness[self.global_best_index]    # self.global_best_fitness = min of private_best_fitness
    
    def update_fitness(self):
        self.fitness = self.environment.get_fitness(self)                               # fresh fitness by pos[d, n]
        mask = self.fitness < self.private_best_fitness                                 # mask = fitness < private_best_fitness
        self.private_best_fitness[mask] = self.fitness[mask]                            # update private_best_fitness
        self.private_best_pos[:, mask] = self.pos[:, mask]
        self.global_best_index = np.argmin(self.private_best_fitness)                   # select best fitness in private_best_fitness
        self.global_best_fitness = self.private_best_fitness[self.global_best_index]    # select best fitness in private_best_fitness
        self.global_best_pos = self.pos[:, self.global_best_index].copy().reshape(self.dimension, 1) # create
        
    def update_vel(self):
        self.v = self.w * self.v + self.c1 * np.random.rand(self.dimension, self.number) * (self.private_best_pos - self.pos) + self.c2 * np.random.rand(self.dimension, self.number) * (self.global_best_pos - self.pos)
        self.v = np.clip(self.v, -self.max_vel, self.max_vel)

    def outputs(self, times=1):
        for _ in tqdm(range(times)):
            self.update_fitness()
            self.update_vel()
            self.pos = self.v + self.pos
        return self.global_best_pos
        
        