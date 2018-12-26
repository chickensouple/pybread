import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def opt_func(vbounce, vball, alpha):
    vpaddle = cvx.Variable(3)
    p = cvx.Variable(3)
    vrel = vball - vpaddle

    constraints = []
    # constraints.append(cvx.norm(p, 2) <= 1)
    # constraints.append(vbounce - vpaddle == alpha * (vrel - 2*p*(p[0]*vrel[0] + p[1]*vrel[1] + p[2]*vrel[2])))

    cost = cvx.norm(vrel)

    problem = cvx.Problem(cvx.Minimize(cost), constraints)
    problem.solve()

    print("Status: " + str(problem.status))
    print("Value: " + str(problem.value))
    print("VPaddle: " + str(vpaddle.value))
    print("p: " + str(p.value))

if __name__ == '__main__':
    # vbounce = np.array([0.1, 0.2, 2.1])
    # vball = np.array([0.3, -0.1, -0.8])
    # alpha = 0.8

    # opt_func(vbounce, vball, alpha)


    p = np.array([0, 0, 1])
    vbounce = np.array([0, 0, 10])
    vball = np.array([1, 1, -5])







# for i in range(10):
#     p = np.random.random_sample(3) * 2 - 1
#     p /= np.linalg.norm(p)
#     if p[2] < 0:
#         p[2] = -p[2]
#     p = np.array([0, 0, 1])

#     vrel = np.random.random_sample(3) * 2 - 1
#     if vrel[2] >= 0:
#         vrel[2] = -vrel[2]
# # p = np.array([0, 0, 1.])
# # vrel = np.array([0.3, 0.1, -0.2])

#     vbounce = vrel - 2*p*np.dot(p, vrel)
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     ax.plot([0, p[0]], [0, p[1]], [0, p[2]], label='p')
#     ax.plot([vrel[0], 0], [vrel[1], 0], [vrel[2], 0], label="vrel")
#     ax.plot([0, vbounce[0]], [0, vbounce[1]], [0, vbounce[2]], label="vbounce")
#     plt.legend()
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.show()


