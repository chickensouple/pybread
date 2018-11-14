from pybread.controllers import LQR
import numpy as np


if __name__ == '__main__':
    A = np.array([[1., 0.1], [0, 1.]])
    B = np.array([[0, 0.1, 0.1], [0.1, 0, 0]])
    Q = np.eye(2)
    P = np.eye(2) * 7
    R = np.eye(3)
    x0 = np.array([[2., 2.]]).T

    lqr = LQR(A, B, P, Q, R)

    N = 10
    batch_controls, batch_cost = lqr.solve_batch(N, x0)
    K_list, P_list = lqr.solve(N, keep_intermediate=True)


    x_state = np.copy(x0)
    recursive_controls = np.zeros((N-1, 3))
    for i in range(N-1):
        K = K_list[-(i+1)]
        u_k = np.dot(K, x_state)
        recursive_controls[i, :] = u_k.squeeze()

        x_state = np.dot(A, x_state) + np.dot(B, u_k)


    recursive_cost = np.dot(x0.T, np.dot(P_list[-1], x0))
    recursive_cost = np.asscalar(recursive_cost)
    print("batch_cost: " + str(batch_cost))
    print("recursive_cost: " + str(recursive_cost))
    print("cost diff: " + str(np.abs(batch_cost - recursive_cost)))


    max_control_err = np.max(np.abs(recursive_controls - batch_controls))
    print("Max control error: " + str(max_control_err))
    print("Max control err / average absolute recursive control: " + str(max_control_err / np.mean(np.abs(recursive_controls))))

