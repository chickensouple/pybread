import numpy as np
import math
from pybread.systems import ModelBase

class SimpleCarModel(ModelBase):
    """
    Models a simple car with front wheel to back wheel of length L
    Car utilizes front wheel steering with max turn angle

    state is x x = [x, y, v, theta]
    where (x, y) is a 2d coordinate, v is the forward velocity of the car
    and theta is the heading of the car (where 0 is pointed in positive x direction)

    control input is u = [a, phi]
    where a is acceleration of the car and phi is turning angle
    """
    def __init__(self, length=1., max_accel=1., max_turn=np.pi/3):
        """
        Initializes a simple car

        Args:
            length (float, optional): front wheel to back wheel distance (default: 1)
            max_accel (float, optional): maximum acceleration of car in m/s/s (default: 1)
            max_turn (TYPE, optional): maximum turn angle of car in rad (default: pi/3)
        """
        control_limits = [np.array([-max_accel, -max_turn]), np.array([max_accel, max_turn])]
        super().__init__(
            state_dim=4, 
            control_dim=2, 
            control_limits=control_limits)
        self.length = length

    def diff_eq(self, x, u):
        u = self._check_and_clip(x, u)

        accel = u[0]
        ang_accel = u[1]

        x_dot = np.zeros(x.shape)
        x_dot[0] = x[2] * math.cos(x[3])
        x_dot[1] = x[2] * math.sin(x[3])
        x_dot[2] = accel
        x_dot[3] = -x[2] / self.length * math.tan(u[1])
        return x_dot
