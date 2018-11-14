import numpy as np
from pybread.systems import ModelBase


class PendulumModel(ModelBase):
    """
    Models a simple pendulum driven by a motor

    state is x = [theta, theta_dot]
    where theta is 0 when pointing upwards
    control input is u = [torque]
    """
    def __init__(self, length=1., mass=0.2, m_type='point', mu=0.05, max_torque=1.):
        control_limits = [np.array([-max_torque]), np.array([max_torque])]
        super().__init__(
            state_dim=2,
            control_dim=1,
            control_limits=control_limits)
        self.length = length
        self.mass = mass
        self.m_type = m_type
        self.mu = mu

        # compute center of mass and moment of inertia
        if m_type == 'point':
            self.com = length
            self.inertia = mass * length * length
        elif m_type == 'rod':
            self.com = 0.5 * length
            self.inertia = mass * length * length / 3
        else:
            raise Exception('Not a valid m_type')

    def diff_eq(self, x, u):
        u = self._check_and_clip(x, u)
        torque = u[0]

        grav = 9.81
        grav_torque = self.mass * grav * self.com * np.sin(x[0])
        fric_torque = -x[1] * self.mu

        x_dot = np.zeros(x.shape)
        x_dot[0] = x[1]
        x_dot[1] = (grav_torque + torque + fric_torque) / self.inertia
        return x_dot

    def after_step(self, x, u):
        # wrap angle to [-pi, pi)
        while x[0] < -np.pi:
            x[0] += 2 * np.pi
        while x[0] >= np.pi:
            x[0] -= 2 * np.pi
        return x

    # extra functions
    def get_energy(self, x):
        kinetic = 0.5 * self.inertia * x[1] * x[1]
        potential = self.mass * 9.81 * self.com * (math.cos(x[0]) - 1)
        total = kinetic + potential
        return total
