import numpy as np
from Environment import Environment
import random

class Pso:
    def __init__(self, number: int, dimension: int, w: float,max_pos:float,  max_vel:float, environment: Environment) -> None:
        self.number = number
        self.dimension = dimension
        self.environment = environment
        self.pos = np.random.uniform(-max_pos, max_pos, (dimension, number))
        self.v = np.random.uniform(-max_vel, max_vel, (dimension, number))
        self.max_vel = max_vel
        self.w = w
        self.c1 = self.c2 = 2
        self.fitness = environment.get_fitness(self.pos)
        self.private_best_fitness = environment.get_fitness(self.pos)
        self.private_best_pos = self.pos.copy()
        self.global_best_index = np.argmin([self.private_best_fitness])
        self.global_best_fitness = self.private_best_fitness[self.global_best_index]
    
    def update_fitness(self):
        self.fitness = self.environment.get_fitness(self.pos)
        for i in range(self.number):
            if self.fitness[i] < self.private_best_fitness[i]:
                self.private_best_pos[0, i], self.private_best_pos[1, i], self.private_best_fitness[i] = self.pos[0, i], self.pos[1, i], self.fitness[i]
        self.global_best_index = np.argmin([self.private_best_fitness])
        self.global_best_fitness = self.private_best_fitness[self.global_best_index]
        
    def update_vel(self):
        self.global_best_pos = np.array([self.pos[i, self.global_best_index] for i in range(self.dimension)]).reshape(self.dimension, 1)
        self.v = self.w * self.v + self.c1 * random.random() * (self.private_best_pos - self.pos) + self.c2 * random.random() * (self.global_best_pos - self.pos)
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
