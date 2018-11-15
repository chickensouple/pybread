import numpy as np
from pybread.systems import ModelBase
import math

#TODO: if time sampling is too large, ball can just penetrate through the paddle

class BouncingBall2DModel(ModelBase):
    """
    Models a ball bouncing in 2d on a paddle
    state is x = [ball_pos_x, ball_pos_y, ball_vel_x, ball_vel_y, paddle_pos_x, paddle_pos_y, paddle_theta]
    paddle_theta = 0 is flat position. positive angle correponds to counterclockwise rotation
    you can move the paddle around and angle it to hit the ball
    control is u = [paddle_vel_x, paddle_vel_y, paddle_angular_vel]

    Attributes:
        max_vel (float, optional): maximum velocity of paddle
        max_ang_vel (float, optional): maximum angular velocity of paddle
        paddle_length (float, optional): length of paddle
        drag_coeff (float, optional): drag coefficient of ball in air (-b * velocity)
        ball_radius (float, optional): radius of ball
        bounce_restitution (float, optional): how much velocity is retained after bounce
    """
    def __init__(self, max_vel=5., max_ang_vel=2., paddle_length=1., drag_coeff=0.05, ball_radius=0.2, bounce_restitution=0.9):
        """
        Initializes Bouncing Ball (2D) Model

        Args:
            max_vel (float, optional): maximum velocity of paddle
            max_ang_vel (float, optional): maximum angular velocity of paddle
            paddle_length (float, optional): length of paddle
            drag_coeff (float, optional): drag coefficient of ball in air (-b * velocity)
            ball_radius (float, optional): radius of ball
            bounce_restitution (float, optional): how much velocity is retained after bounce
        """
        self.max_vel = max_vel
        self.max_ang_vel = max_ang_vel

        self.paddle_length = paddle_length
        self.drag_coeff = drag_coeff
        self.ball_radius = ball_radius
        self.bounce_restitution = bounce_restitution

        # maximum velocity is handled in diff_eq() as it is not a box constraint
        super().__init__(
            state_dim=7,
            control_dim=3,
            control_limits=[np.array([-np.inf, -np.inf, -max_ang_vel]), np.array([np.inf, np.inf, max_ang_vel])])


    def diff_eq(self, x, u):
        u = self._check_and_clip(x, u)
        vel_mag = np.linalg.norm(u[:2])
        if vel_mag > self.max_vel:
            u[:2] *= (self.max_vel / vel_mag)

        x_dot = np.zeros(x.shape)
        # ball states
        x_dot[0] = x[2]
        x_dot[1] = x[3]
        x_dot[2] = -self.drag_coeff * x[2]
        x_dot[3] = -self.drag_coeff * x[3] - 9.81

        # paddle states
        x_dot[4] = u[0]
        x_dot[5] = u[1]
        x_dot[6] = u[2]

        return x_dot

    def after_step(self, x, u):
        parallel_vec = np.array([math.cos(x[6]), math.sin(x[6])])
        dist_vec = x[0:2] - x[4:6]
        dist_mag = np.linalg.norm(dist_vec)

        parallel_dist = np.abs(np.dot(parallel_vec, dist_vec))

        # find vector perpendicular to paddle and pointing towards ball
        perpendicular_vec1 = np.array([parallel_vec[1], -parallel_vec[0]])
        perpendicular_vec2 = np.array([-parallel_vec[1], parallel_vec[0]])

        if np.linalg.norm(perpendicular_vec1) < np.linalg.norm(perpendicular_vec2):
            perpendicular_vec = perpendicular_vec1
        else:
            perpendicular_vec = perpendicular_vec2

        perpendicular_dist = np.abs(np.dot(perpendicular_vec, dist_vec))

        print("perpendicular_dist: " + str(perpendicular_dist))
        if (perpendicular_dist > self.ball_radius or parallel_dist > self.paddle_length*0.5):
            return x

        perpendicular_vec /= np.linalg.norm(perpendicular_vec)

        # compute total vel of paddle point from the angular velocity
        vel_turning = -perpendicular_vec * (parallel_dist * u[2])
        vel_paddle = vel_turning + u[0:2]

        # compute relative velocity of ball to paddle
        vel_rel = x[2:4] - vel_paddle
        vel_rel_mag = np.linalg.norm(vel_rel)


        if vel_rel_mag <= 1e-3:
            # if relative velocity is too small, just set velocity of ball to be the same
            # the paddle's
            x[2:4] = vel_paddle
            return x


        vel_reflect = vel_rel - 2 * np.dot(vel_rel, perpendicular_vec) * perpendicular_vec
        x[2:4] = vel_reflect * self.bounce_restitution

        return x





