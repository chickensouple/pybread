import numpy as np
from pybread.systems import ModelBase

class LTISystemModel(ModelBase):
    """
    Generic Linear Time Invariant System of the form
    x_dot = Ax + Bu
    """
    def __init__(self, A, B, control_limits=None):
        if A.shape[0] != B.shape[0]:
            raise Exception('Number of states do not match')

        state_dim = A.shape[0]
        control_dim = B.shape[1]
        if control_limits == None:
            min_limit = np.ones(control_dim) * -np.Inf
            max_limit = np.ones(control_dim) * np.Inf
            control_limits = [min_limit, max_limit]
        super().__init__(
            state_dim=state_dim, 
            control_dim=control_dim, 
            control_limits=control_limits)

        self.A = A
        self.B = B

    def diff_eq(self, x, u):
        u = self._check_and_clip(x, u)
        x_dot = np.dot(self.A, x) + np.dot(self.B, u)
        return x_dot

    def linearize(self, x0, u0):
        f0 = self.diff_eq(x0, u0)
        return f0, self.A, self.B
