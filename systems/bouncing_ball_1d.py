import numpy as np
from pybread.systems import ModelBase

class BouncingBall1DModel(ModelBase):
    """
    Models a ball bouncing in 1d on a paddle
    state is x = [ball_pos, ball_vel, paddle_pos]
    pos stands for position, vel for velocity
    ball should start with a higher position than paddle. Gravity applies negative acceleration
    control is u = [paddle velocity]

    Attributes:
        ball_radius (float): radius of ball
        bounce_restitution (float): how much of velocity is retained after bounce
        drag_coeff (float): drag of ball in air
        max_vel (float): maximum acceleration of the paddle
    """
    def __init__(self, max_vel=5., drag_coeff=0.05, ball_radius=0.2, bounce_restitution=0.9):
        """
        Initializes Bouncing Ball (1D) Model

        Args:
            max_vel (float, optional): maximum velocity of paddle
            drag_coeff (float, optional): drag coefficient of ball in air (-b * velocity)
            ball_radius (float, optional): radius of ball
            bounce_restitution (float, optional): how much velocity is retained after bounce
        """
        self.max_vel = max_vel

        self.drag_coeff = drag_coeff
        self.ball_radius = ball_radius
        self.bounce_restitution = bounce_restitution

        super().__init__(
            state_dim=3,
            control_dim=1,
            control_limits=[np.array([-self.max_vel]), np.array([self.max_vel])])

    def diff_eq(self, x, u):
        u = self._check_and_clip(x, u)
        x_dot = np.zeros(x.shape)
        x_dot[0] = x[1]
        x_dot[2] = u[0]

        x_dot[1] = -(x[1] * self.drag_coeff) - (9.81)

        return x_dot

    def after_step(self, x, u):
        delta_pos = x[0] - x[2]
        if (delta_pos > self.ball_radius):
            return x

        # if ball is touching or penetrating paddle, make the ball jump to surface of paddle and
        # reverse the velocity 
        if x[0] >= x[2]:
            x[0] = x[2] + self.ball_radius
        else:
            x[0] = x[2] - self.ball_radius
        x[1] = -(x[1] - u[0]) * self.bounce_restitution
        return x
