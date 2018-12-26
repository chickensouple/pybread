import numpy as np
import numpy.linalg
import math



class BouncingBall1DController(object):
    def __init__(self, coeff_of_restitution, desired_height):
        self.coeff_of_restitution = coeff_of_restitution
        self.desired_height = desired_height

    def get_control(self, x):
        grav = -9.81
        time_until_impact = (-x[1] - np.sqrt(x[1]**2 - 2*grav*x[0])) / grav
        ball_vel_at_impact = x[1] + grav*time_until_impact

        desired_vel_after_impact = np.sqrt(2*-grav*self.desired_height)
        desired_paddle_vel = (desired_vel_after_impact + ball_vel_at_impact*self.coeff_of_restitution) / (self.coeff_of_restitution + 1)


        # fit a quadratic path to the controller
        tau = time_until_impact
        coeffs = np.zeros(3) # from highest ordered coeff to lowest
        A = np.array([[tau**2, tau], [2*tau, 1.]])
        b = np.array([[0., desired_paddle_vel]]).T - np.array([[x[2], 0]]).T
        sol = np.linalg.solve(A, b).squeeze()
        coeffs[0] = sol[0]
        coeffs[1] = sol[1]
        coeffs[2] = x[2]

        # return the current desired velocity
        current_vel = coeffs[1]

        action = np.array([current_vel])
        return action


