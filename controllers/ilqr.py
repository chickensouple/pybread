import numpy as np
import math

#TODO: make all dimensions consistent with models!

class iLQR(object):
    def __init__(self, model):
        self.model = model

    def solve(self, x0, N, dt, quad_step_cost_func, quad_final_cost_func, niter=50):
        n = len(x0)
        m = 1

        controller_list = [(np.zeros((m, n)), np.zeros((m, 1))) for _ in range(N)]
        x_list = [np.zeros((n, 1)) for _ in range(N)]
        x_list[0] = x0
        u_list = [np.zeros((m, 1)) for _ in range(N)]

        for i in range(niter):
            x_list, u_list = self._forward_pass(x_list, u_list, controller_list, dt)
            controller_list = self._backward_pass(x_list, u_list, dt, quad_step_cost_func, quad_final_cost_func)

            # import matplotlib.pyplot as plt
            # plt.figure(2)
            # x_arr = np.array(x_list)
            # t = np.ones(N+1) * dt
            # t = np.cumsum(t)
            # plt.cla()
            # plt.plot(t, x_arr[:, 0], label='theta', c='b')
            # plt.plot(t, x_arr[:, 1], label='theta_dot', c='r')
            # plt.legend()
            # plt.show(block=False)
            # plt.pause(0.01)
            # # input('Press Enter to Continue: ')


        return controller_list, x_list, u_list

    def _forward_pass(self, x_bar_list, u_bar_list, controller_list, dt):
        x_list = [x_bar_list[0]]
        u_list = []
        for i, (x_bar, u_bar, controller) in enumerate(zip(x_bar_list, u_bar_list, controller_list)):
            x = x_list[-1]
            u = np.dot(controller[0], x - x_bar) + controller[1] + u_bar


            u = np.reshape(u, (self.model.control_dim))
            x = np.reshape(x, (self.model.state_dim))

            x_new = self.model.get_next_state(x, u, dt)
            x_new = np.reshape(x_new, (self.model.state_dim, 1))

            u = np.reshape(u, (self.model.control_dim, 1))


            u_list.append(u)
            x_list.append(x_new)

        return x_list, u_list


    def _backward_pass(self, x_list, u_list, dt, quad_step_cost_func, quad_final_cost_func):
        controller_list = []

        f0, b, A = quad_final_cost_func(x_list[-1])

        for x, u in zip(reversed(x_list), reversed(u_list)):

            x_squeeze = np.reshape(x, (self.model.state_dim))
            u_squeeze = np.reshape(u, (self.model.control_dim))
            f0, A_sys, b_sys = self.model.linearize(x_squeeze, u_squeeze)

            # discretize our linearized system
            f_x = np.eye(A_sys.shape[0]) + A_sys * dt
            f_u = b_sys * dt

            # quadraticize costs
            f0, lx, lu, lxx, luu, lux = quad_step_cost_func(x, u)

            qx = lx + np.dot(f_x.T, b)
            qu = lu + np.dot(f_u.T, b)
            qxx = lxx + np.dot(f_x.T, np.dot(A, f_x))
            quu = luu + np.dot(f_u.T, np.dot(A, f_u))
            qux = lux + np.dot(f_u.T, np.dot(A, f_x))

            K = -np.linalg.solve(quu, qux)
            j = -np.linalg.solve(quu, qu)
            A = qxx + np.dot(K.T, np.dot(quu, K)) + np.dot(qux.T, K) + np.dot(K.T, qux)
            b = qx + np.dot(K.T, np.dot(quu, j)) + np.dot(qux.T, j) + np.dot(K.T, qu)

            controller_list.append((K, j))

        controller_list.reverse()
        return controller_list

