import numpy as np
from pybread.systems import ModelBase

class RocketModel(ModelBase):
    """
    Models a simple rocket

    state is [h, v, m]
    where h is height, v is velocity, m is mass of fuel
    control input is [rate of mass explusion] in mass / s. needs to be negative
    """
    def __init__(self, ve=150., max_u=10., mass_r=1.):
        """
        Initializes a simple rocket

        Args:
            ve (float, optional): magnitude of expulsion velocity (positive number)
            max_u (float, optional): magnitude of maximum mass expulsion rate (positive number)
            mass_r (float, optional): mass of rocket not counting fuel
        """
        control_limits = [np.array([-max_u]), np.array([0])]
        super().__init__(
            state_dim=3, 
            control_dim=1, 
            control_limits=control_limits)
        self.ve = -ve
        self.mass_r = mass_r

    def diff_eq(self, x, u):
        u = self._check_and_clip(x, u)

        x_dot = np.zeros(x.shape)
        x_dot[0] = x[1]

        # if there is no more fuel, don't
        # add acceleration from fuel explusion
        if x[2] <= 0:
            x_dot[1] = -9.81
            x_dot[2] = 0
        else:
            x_dot[1] = -9.81 + u * self.ve / (x[2] + self.mass_r)
            x_dot[2] = u

        return x_dot

    def after_step(self, x, u):
        if x[0] < 0:
            x[0] = 0
            x[1] = 0
        return x
