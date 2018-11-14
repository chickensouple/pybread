from pybread.controllers import iLQR
from pybread.systems import PendulumModel
import numpy as np
from functools import partial
import math
import matplotlib.pyplot as plt

if __name__ == '__main__':
    Cx = np.array([[100., 0], [0, 1]])
    Cu = np.array([[490.]])
    # Cu = np.array([[0.01]])

    def final_cost(x, target_state, Cx=Cx):
        delta_x = x - target_state
        cost = np.dot(delta_x.T, np.dot(Cx, delta_x))
        return cost

    def quad_final_cost(x, target_state, Cx=Cx):
        A = 2*Cx
        b = 2*np.dot(Cx, x - target_state)
        f0 = final_cost(x, target_state)
        return f0, b, A

    def step_cost(x, u, target_state, target_control, Cx=Cx, Cu=Cu):
        delta_x = x - target_state
        delta_u = u - target_control

        cost = np.dot(delta_x.T, np.dot(Cx, delta_x))

        cost += np.dot(delta_u.T, np.dot(Cu, delta_u))
        return cost 

    def quad_step_cost(x, u, target_state, target_control, Cx=Cx, Cu=Cu):
        delta_x = x - target_state
        delta_u = u - target_control

        lx = 2*np.dot(Cx, delta_x)
        lxx = 2*Cx
        lu = 2*np.dot(Cu, delta_u)
        luu = 2*Cu

        m = len(u)
        n = len(x)
        lux = np.zeros((m, n))
        f0 = step_cost(x, u, target_state, target_control)
        return f0, lx, lu, lxx, luu, lux



    target_state = np.array([[0, 0.]]).T
    target_control = np.array([[0.]])
    step_cost_func = partial(step_cost, target_state=target_state, target_control=target_control)
    quad_step_cost_func = partial(quad_step_cost, target_state=target_state, target_control=target_control)

    final_cost_func = partial(final_cost, target_state=target_state)
    quad_final_cost_func = partial(quad_final_cost, target_state=target_state)

    dt = 0.05
    x0 = np.array([[math.pi, 0.]]).T
    pendulum = PendulumModel(max_torque=np.Inf)

    ilqr = iLQR(pendulum)

    N = 500
    controller_list, x_list, u_list = ilqr.solve(x0, N, dt, quad_step_cost_func, quad_final_cost_func)


    curr_state = x0
    states = np.zeros((2, N-1))
    controls = np.zeros(N-1)
    for i in range(N-1):
        states[:, i] = x0.squeeze()

        controller = controller_list[0]
        u = np.dot(controller[0], curr_state - x_list[0]) + controller[1] + u_list[0]

        curr_state = pendulum.get_next_state(np.reshape(curr_state, (pendulum.state_dim)),
            np.reshape(u, (pendulum.control_dim)), 
            dt)
        curr_state = np.reshape(curr_state, (pendulum.state_dim, 1))
        controls[i] = u.squeeze()

    plt.figure(1)
    plt.cla()
    t = np.ones(N-1) * dt
    t = np.cumsum(t)
    plt.plot(t, states[0, :], label='theta')
    plt.plot(t, states[1, :], label='theta_dot')
    plt.plot(t, controls, label='control')
    plt.legend()

    plt.figure(2)
    plt.cla()
    # plotting original ddp solution
    t = np.ones(N) * dt
    t = np.cumsum(t)
    x_arr = np.array(x_list).squeeze()
    plt.plot(t, x_arr[:-1, 0], label='theta')
    plt.plot(t, x_arr[:-1, 1], label='theta_dot')
    plt.plot(t, np.array(u_list).squeeze(), label='control')
    plt.legend()
    plt.show()
