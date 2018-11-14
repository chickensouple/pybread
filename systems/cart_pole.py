import numpy as np
from pybread.systems import ModelBase

class CartPoleModel(ModelBase):
    """
    Models a typical cartpole system with a pointmass pendulum

    state is [x, theta, x_dot, theta_dot]
    where x is location on track
    theta is angle of pendulum (0 is pointing downwards)
    """
    def __init__(self, mass_cart=1., length=1., mass_pendulum=0.2, max_force=1.):
        """
        Initializes a Cart Pole

        Args:
            mass_cart (float, optional): mass of cart in kg
            length (float, optional): length of pendulum in m
            mass_pendulum (float, optional): mass of pendulum in kg
            max_force (float, optional): maximum force applicable in N
            **kwargs: Description
        """
        control_limits = [np.array([-max_force]), np.array([max_force])]
        super().__init__(
            state_dim=4, 
            control_dim=1, 
            control_limits=control_limits)
        self.mass_cart = mass_cart
        self.length = length
        self.mass_pendulum = mass_pendulum

    def diff_eq(self, x, u):
        u = self._check_and_clip(x, u)
        force = u[0]

        x_dot = np.zeros(x.shape)
        x_dot[0] = x[2]
        x_dot[1] = x[3]

        sin_theta = np.sin(x[1])
        cos_theta = np.cos(x[1])
        grav = 9.81
        den = 1. / (self.mass_cart + self.mass_pendulum * np.square(sin_theta))
        x_dot[2] = force + self.mass_pendulum * sin_theta * (self.length * np.square(x[3]) + grav * cos_theta)
        x_dot[2] *= den
        x_dot[3] = -(force*cos_theta + \
            self.mass_pendulum*self.length*np.square(x[3])*cos_theta*sin_theta + \
            (self.mass_cart+self.mass_pendulum)*grav*sin_theta)
        x_dot[3] *= den
        x_dot[3] /= self.length
        return x_dot

    def after_step(self, x, u):
        # wrap angle to [-pi, pi)
        while x[1] < -np.pi:
            x[1] += 2 * np.pi
        while x[1] >= np.pi:
            x[1] -= 2 * np.pi
        return x
