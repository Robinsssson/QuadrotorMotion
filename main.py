import numpy as np
import matplotlib.pyplot as plt
from Environment.Environment import Environment
from Model.PSO.Pso import Pso

class MyTest(Environment):
    def __init__(self, size: int):
        super(MyTest, self).__init__(size)

    def get_fitness(self, pos):
        return (pos[0] - 12) ** 2 + (pos[1] - 9) ** 2

def main():
    np.random.seed(10)
    fig = plt.figure(figsize=(24, 10))
    ax1 = fig.add_subplot(141)
    ax2 = fig.add_subplot(142)
    ax3 = fig.add_subplot(143)
    ax4 = fig.add_subplot(144)
    envi = MyTest(0)
    n = 5000
    model = Pso(n, 2, 0.5, 1000, 5, envi)
    # ax.scatter(model.pos[0], model.pos[1])
    ax1.scatter(model.pos[0], model.pos[1])
    model.outputs(100)
    ax2.scatter(model.pos[0], model.pos[1])
    model.outputs(100)
    ax3.scatter(model.pos[0], model.pos[1])
    model.outputs(100)
    ax4.scatter(model.pos[0], model.pos[1])
    print(
        f"{model.pos[0, model.global_best_index]}, \
        {model.pos[1, model.global_best_index]}"
    )
    print(model.global_best_fitness)
    # Plot some data on the axes.
    plt.show()


if __name__ == "__main__":
    main()
