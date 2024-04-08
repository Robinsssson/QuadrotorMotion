import numpy as np
from Environment import Environment
import random

class Pso:
    def __init__(self, number: int, dimension: int, w: float,max_pos:float,  max_vel:float, environment: Environment) -> None:
        self.number = number                     #设置私有数目
        self.dimension = dimension               #设置变量维度
        self.environment = environment           #添加场地条件
        self.pos = np.random.uniform(-max_pos, max_pos, (dimension, number))     #通过max_pos 创建pos[d, n]
        self.v = np.random.uniform(-max_vel, max_vel, (dimension, number))       #using max_vec. create v[d, n]
        self.max_vel = max_vel
        self.w = w
        self.c1 = self.c2 = 2
        self.fitness = environment.get_fitness(self.pos)                        
        self.private_best_fitness = environment.get_fitness(self.pos)                   #private_best_fitness[n] = function(pos)
        self.private_best_pos = self.pos.copy()                                         #default make private_best_pos[d, n] = pos[d, n]
        self.global_best_index = np.argmin([self.private_best_fitness])                 #self.global_best_index = index of min private_best_fitness
        self.global_best_fitness = self.private_best_fitness[self.global_best_index]    #self.global_best_fitness = min of private_best_fitness
    
    def update_fitness(self):
        self.fitness = self.environment.get_fitness(self.pos)                           
        for i in range(self.number):
            if self.fitness[i] < self.private_best_fitness[i]:
                self.private_best_fitness[i] = self.fitness[i]
                for j in range(self.dimension):
                    self.private_best_pos[j, i] = self.pos[j, i]
        self.global_best_index = np.argmin([self.private_best_fitness])
        self.global_best_fitness = self.private_best_fitness[self.global_best_index]
        
    def update_vel(self):
        self.global_best_pos = np.array([self.pos[i, self.global_best_index] for i in range(self.dimension)]).reshape(self.dimension, 1)
        self.v = self.w * self.v + self.c1   * random.random() * (self.private_best_pos - self.pos) + self.c2 * random.random() * (self.global_best_pos - self.pos)
        for vi in self.v:
            for v in vi:
                if v >= self.max_vel:
                    v = self.max_vel
                if v <= -self.max_vel:
                    v = -self.max_vel

    def outputs(self, times=1):
        for _ in range(times):
            self.update_vel()
            self.pos = self.v + self.pos
            self.update_fitness()
