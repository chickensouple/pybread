import numpy as np
import math
from pybread.controllers.bouncing_ball_1d_controller import *
from pybread.systems import BouncingBall1DModel
import matplotlib.pyplot as plt

if __name__ == '__main__':
    restitution = 0.9
    desired_height = 7

    model = BouncingBall1DModel(max_vel=float('inf'), drag_coeff=0., bounce_restitution=restitution)
    controller = BouncingBall1DController(restitution, desired_height)

    dt = 0.01

    T = 2000
    state_hist = np.zeros((T, 3))
    control_hist = np.zeros((T, 1))

    state = np.array([5., 0., -2])
    for t in range(T):
        action = controller.get_control(state)
        new_state = model.get_next_state(state, action, dt)

        state_hist[t, :] = state
        control_hist[t, :] = action


        state = new_state



    time_array = [dt*i for i in range(T)]

    plt.subplot(3, 1, 1)
    plt.plot(time_array, state_hist[:, 2], label='paddle')
    plt.plot(time_array, state_hist[:, 0], label='ball')
    plt.plot(time_array, desired_height * np.ones(len(time_array)), label='desired')
    plt.legend()
    plt.xlabel("time (s)")
    plt.ylabel("pos (m)")



    plt.subplot(3, 1, 2)
    plt.plot(time_array, state_hist[:, 1])
    plt.xlabel("time (s)")
    plt.ylabel("ball vel (m/s)")


    plt.subplot(3, 1, 3)
    plt.plot(time_array, control_hist[:, 0])
    plt.xlabel("time (s)")
    plt.ylabel("paddle vel (m/s)")

    plt.show()


