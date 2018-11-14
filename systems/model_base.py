import numpy as np
import math
import scipy.integrate

class ModelBase(object):
    """
    Base Class for Models that are described by
    Ordinary Time Invariant Differential Equations, x_dot = f(x, u)
    All states are numpy arrays of size state_dim
    All control inputs are numpy arrays of size control_dim

    This can also model functions with more complicated interactions (contacts)
    by using the after_step() function, which can be used to modify the state after one step of
    the ode solver

    Attributes:
        state_dim (int): number of state dimensions
        control_dim (int): number of control dimensions
        control_limits (list of 2 numpy arrays): lower and upper limits of control
        control_limis[0] is lower bound on control inputs
        control_limits[1] is upper bound on control inputs
        if control_limits is None, there are no control limits
    """
    def __init__(self, state_dim, control_dim, control_limits=None):
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.control_limits = control_limits

    def diff_eq(self, x, u):
        """
        Calculates the derivative, f, for a batch of size N of state and control inputs.
        So for a single state and control input, use N=1

        Args:
            x (numpy array): size state_dim array of state
            u (numpy array): size control_dim array of control
        """
        raise Exception("Needs to be implemented")

    def after_step(self, x, u):
        """
        Can be overloaded to modify state, x, after one step of ode solver

        Args:
            x (numpy array): size (state_dim) array of state
            u (numpy array): size (control_dim) array of control

        Returns:
            (numpy array): new state of size (state_dim)
        """
        return x

    def get_next_state(self, x, u, dt):
        """
        Gets the new state of the system after starting
        in state x, and applying a control input u for a time dt

        Args:
            x (numpy array): size (state_dim) array of state
            u (numpy array): size (control_dim) array of control
            dt (float): time step

        Returns:
            (numpy array): size (state_dim) array for the new state
        """
        result = scipy.integrate.odeint(
            func=self._ode_func,
            y0=x,
            t=[0, dt],
            args=(u,))

        # get rid of first time step
        result = result[1]

        result = self.after_step(result, u)
        return result

    def get_traj(self, x, u, dt):
        """
        Gets a trajectory of the system starting at state x,
        with a sequence of control inputs, each applied for a time dt

        Args:
            x (numpy array): size (state_dim) array of state
            u (numpy array): size (N by control_dim) array of controls
            each row corresponds to the control at a time step
            dt (float): time step

        Returns:
            numpy array: size (N+1 by state_dim) of the trajectory
            first row is the initial state
        """
        traj = np.zeros((len(u)+1, self.state_dim))
        traj[0, :] = x
        for i in range(len(u)):
            traj[i+1, :] = self.get_next_state(traj[i, :], u[i, :], dt)
        return traj

    def linearize(self, x0, u0):
        """
        Linearize the model around (x0, u0).
        Default behaviour will perform numerical linearization with finite differences,
        but this can be overridden with specific analytic linearizations

        f(x) ~= f0 + A*(dx) + B*(du)

        Args:
            x0 (numpy arr): state to linearize around. size (state_dim) array
            u0 (numpy arr): control to linearize around. size (control_dim) array

        Returns:
            (list of numpy arrs): (f0, A, B)
        """
        f0 = self.diff_eq(x0, u0)

        dx = 0.01
        du = 0.01

        A = np.zeros((self.state_dim, self.state_dim))
        B = np.zeros((self.state_dim, self.control_dim))

        for i in range(self.state_dim):
            vec_dx = np.zeros(self.state_dim)
            vec_dx[i] = dx
            new_f_x = self.diff_eq(x0 + vec_dx, u0)
            delta_f_x = (new_f_x - f0) / dx
            A[:, i] = delta_f_x
        for i in range(self.control_dim):
            vec_du = np.zeros(self.control_dim)
            vec_du[i] = du
            new_f_u = self.diff_eq(x0, u0 + vec_du)
            delta_f_u = (new_f_u - f0) / du
            B[:, i] = delta_f_u

        return f0, A, B


    def _check_and_clip(self, x, u):
        """
        Helper function for checking if state and control dimensions are right
        and clipping control as necessary

        Checks control and state shapes
        Checks a control input against control limits
        and clip the control input if needed

        Args:
            x (numpy arr): state
            u (numpy arr): control

        Returns:
            numpy arr: clipped control
        """
        if len(x) != self.state_dim:
            raise Exception("State is not proper shape")
        if len(u) != self.control_dim:
            raise Exception("Control input is not proper shape")

        if self.control_limits == None:
            return u

        u = np.clip(u, self.control_limits[0], self.control_limits[1])
        return u

    def _ode_func(self, x, t, u):
        """
        Internal function to use for scipy's ode integration interface

        Args:
            x (numpy array): size state_dim numpy array
            t (float): time
            u (numpy array): size control_dim numpy array

        Returns:
            numpy array: derivative of state
        """
        return self.diff_eq(x, u)
