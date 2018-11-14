import numpy as np
from pybread.systems import ModelBase
from pybread.systems import LTISystemModel

class DoubleIntegratorModel(LTISystemModel):
    """
    Models a simple double integrator

    state for system is [x, x_dot]
    control input is [acceleration]
    """
    def __init__(self, max_acc=1.):
        """
        Initializes a Double Integrator

        Args:
            max_acc (float, optional): maximum acceleration in m/s/s
        """
        A = np.array([[0., 1.], [0., 0.]])
        B = np.array([[0., 1.]]).T
        control_limits = [np.array([-max_acc]), np.array([max_acc])]
        super().__init__(A, B, control_limits)


