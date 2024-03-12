import numpy as np
import scipy as sp

# status outputs


#Singleton class
class quadrotor:

    def __init__(self, mass: float, l, b, d, *, i_array: list) -> None:
        self.mass = mass
        self.l = l
        self.b = b
        self.d = d
        self.i_array = np.asarray(i_array, dtype=float)
        self.u_array = np.array([0, 0, 0, 0])
        self.alpha_acceleration_array = np.array([0, 0, 0])

    def status(self, w1, w2, w3, w4) -> None:
        self.w_array = np.array([w1, w2, w3, w4])
        self.u_array[0] = np.sum(self.w_array**2) * self.b
        self.u_array[1] = (w4**2 - w2**2) * self.b
        self.u_array[2] = (w3**2 - w1**2) * self.b
        self.u_array[3] = (w1**2 + w3**2 - w2**2 - w4**2) * self.d

    def alpha_acceleration_outputs(self) -> np.array:
        self.alpha_acceleration_array = np.array([
            self.l * self.u_array[1] / self.i_array[0],
            self.l * self.u_array[2] / self.i_array[1],
            self.u_array[3] / self.i_array[2]
        ])
        return self.alpha_acceleration_array.copy()
