import numpy as np
from pybread.systems import ModelBase

class BouncingBallModel(ModelBase):

    """
    Models a ball bouncing in 1d on a paddle
    state is x = [pos_ball, vel_ball, pos_paddle, vel_paddle]
    pos stands for position, vel for velocity
    ball should start with a higher position than paddle. Gravity applies negative acceleration

    Attributes:
        ball_radius (float): radius of ball
        bounce_restitution (float): how much of velocity is retained after bounce
        drag_coeff (float): drag of ball in air
        max_acc (float): maximum acceleration of the paddle
    """
    def __init__(self, max_acc=1., drag_coeff=0.05, ball_radius=0.1, bounce_restitution=0.9):
        """
        Initializes Bouncing Ball Model

        Args:
            max_acc (float, optional): maximum acceleration of paddle
            drag_coeff (float, optional): drag coefficient of ball in air (-b * velocity)
            ball_radius (float, optional): radius of ball
            bounce_restitution (float, optional): how much velocity is retained after bounce
        """
        self.max_acc = max_acc

        self.drag_coeff = drag_coeff
        self.ball_radius = ball_radius
        self.bounce_restitution = bounce_restitution

        super().__init__(
            state_dim=4,
            control_dim=1,
            control_limits=[np.array([-self.max_acc]), np.array([self.max_acc])])

    def diff_eq(self, x, u):
        self._check_and_clip(x, u)
        x_dot = np.zeros(x.shape)
        x_dot[0] = x[1]
        x_dot[2] = x[3]

        x_dot[3] = u[0]
        x_dot[1] = -(x[1] * self.drag_coeff) - (9.81)

        return x_dot

    def after_step(self, x, u):
        delta_pos = x[0] - x[2]
        # if ball is touching or penetrating paddle, make the ball jump to surface of paddle and
        # reverse the velocity 
        if (delta_pos <= self.ball_radius):
            x[0] = x[2] + self.ball_radius
            x[1] = -(x[1] + x[3]) * self.bounce_restitution
        return x
