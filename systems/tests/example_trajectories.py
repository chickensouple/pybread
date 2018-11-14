from pybread.systems import *
import numpy as np
import matplotlib.pyplot as plt
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Basic examples of trajectories of different models. \
        Use --model flag to identify which model")
    parser.add_argument("--model", 
        dest="model",
        type=str, 
        required=True,
        choices=["pendulum", "bouncing_ball", "lti", 
            "double_integrator", "dubins_car", "simple_car", 
            "cart_pole", "rocket"],
        help="Choose which model to test with.")

    args = parser.parse_args()

    if args.model == "pendulum":
        model = PendulumModel()

        state = np.array([0.1, 0])
        control = np.array([0.2])
        dt = 0.05

        num_time_steps = 200
        controls = np.tile(control, [num_time_steps, 1])
        traj = model.get_traj(state, controls, dt)

        t = [i*dt for i in range(len(controls)+1)]

        plt.figure(1)
        theta = traj[:, 0]
        theta[theta < 0] += 2 * np.pi
        plt.plot(theta, traj[:, 1])
        plt.xlabel("theta (rad)")
        plt.ylabel("theta_dot (rad/s)")

        plt.figure(2)
        plt.subplot(2, 1, 1)
        plt.plot(t, theta)
        plt.xlabel("time (s)")
        plt.ylabel("theta (rad)")

        plt.subplot(2, 1, 2)
        plt.plot(t, traj[:, 1])
        plt.xlabel("time (s)")
        plt.ylabel("theta_dot (rad/2)")
        plt.show()

    elif args.model == "bouncing_ball":
        model = BouncingBallModel()

        state = np.array([5, 0, 1, 0])
        control = np.array([0.1])
        dt = 0.05

        num_time_steps = 100
        controls = np.tile(control, [num_time_steps, 1])

        traj = model.get_traj(state, controls, dt)
        t = [i*dt for i in range(len(controls)+1)]

        plt.subplot(4, 1, 1)
        plt.plot(t, traj[:, 0])
        plt.xlabel("time (s)")
        plt.ylabel("ball pos (m)")

        plt.subplot(4, 1, 2)
        plt.plot(t, traj[:, 1])
        plt.xlabel("time (s)")
        plt.ylabel("ball vel (m/s)")


        plt.subplot(4, 1, 3)
        plt.plot(t, traj[:, 2])
        plt.xlabel("time (s)")
        plt.ylabel("paddle pos (m)")

        plt.subplot(4, 1, 4)
        plt.plot(t, traj[:, 3])
        plt.xlabel("time (s)")
        plt.ylabel("paddle vel (m/s)")

        plt.show()
    elif args.model == "lti":
        A = np.array([[-0.4, 1.], [-1., 0.2]])
        B = np.zeros((2, 1))
        model = LTISystemModel(A, B)

        control = np.array([0.])

        num_time_steps = 1000
        controls = np.tile(control, [num_time_steps, 1])
        dt = 0.05
        state = np.array([1, 1])

        traj = model.get_traj(state, controls, dt)
        color = [[float(i) / len(traj), 0., 1.] for i in range(len(traj))]
        plt.scatter(traj[:, 0], traj[:, 1], color=color)
        plt.xlabel("x1")
        plt.ylabel("x2")
        print("2 state system. Plotting trajectory in statespace. Blue is initial state. Pink is final.")
        plt.show()
    elif args.model == "double_integrator":
        model = DoubleIntegratorModel()

        state = np.array([-1, -5])
        control = np.array([1.])
        dt = 0.05

        num_time_steps = 100
        controls = np.tile(control, [num_time_steps, 1])
        traj = model.get_traj(state, controls, dt)
        plt.plot(traj[:, 0], traj[:, 1])
        plt.xlabel("pos (m)")
        plt.ylabel("vel (m/s)")
        plt.show()
    elif args.model == "dubins_car":
        model = DubinsCarModel()

        state = np.array([0, 0, 0.])
        control = np.array([[1., 1., 1., -1., -1, -1]]).T
        dt = 0.1

        num_time_steps = 5
        controls = np.tile(control, [num_time_steps, 1])
        traj = model.get_traj(state, controls, dt)
        plt.plot(traj[:, 0], traj[:, 1])
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.show()
    elif args.model == "simple_car":
        model = SimpleCarModel()
        state = np.array([0, 0, 0, 0.])
        dt = 0.1

        controls = \
            np.array([[1., 0],
                      [1., 0],
                      [1., 0.5],
                      [1., 0.5],
                      [0, 0.5],
                      [0, 0.5],
                      [0, 0.5],
                      [0, 0.5],
                      [-1, 0],
                      [-1, 0],
                      [-1, 0],
                      [-1, 0],
                      [0, 0]])
        num_time_steps = len(controls)
        traj = model.get_traj(state, controls, dt)
        plt.plot(traj[:, 0], traj[:, 1])
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.show()
    elif args.model == "cart_pole":
        model = CartPoleModel()
        state = np.array([0, 0, 0, 0.])
        control = np.array([1.])
        dt = 0.05

        num_time_steps = 30
        controls = np.tile(control, [num_time_steps, 1])
        traj = model.get_traj(state, controls, dt)

        plt.figure(1)
        color = [[float(i) / len(traj), 0., 1.] for i in range(len(traj))]
        plt.scatter(traj[:, 1], traj[:, 3], color=color)
        plt.xlabel("theta (rad)")
        plt.ylabel("theta_dot (rad/s)")

        plt.figure(2)
        t = [i*dt for i in range(len(controls)+1)]
        plt.plot(t, traj[:, 0])
        plt.xlabel("time (s)")
        plt.ylabel("position (m)")

        plt.show()
    elif args.model == "rocket":
        model = RocketModel(ve=50.)
        state = np.array([0, 0, 10.])
        control = np.array([-10.])
        dt = 0.1

        num_time_steps = 300
        controls = np.tile(control, [num_time_steps, 1])
        traj = model.get_traj(state, controls, dt)


        t = [i*dt for i in range(len(controls)+1)]
        plt.subplot(3, 1, 1)
        plt.plot(t, traj[:, 0], label='height')
        plt.xlabel("time (s)")
        plt.ylabel("height (m)")

        plt.subplot(3, 1, 2)
        plt.plot(t, traj[:, 1], label='velocity')
        plt.xlabel("time (s)")
        plt.ylabel("velocity (m/s)")

        plt.subplot(3, 1, 3)
        plt.plot(t, traj[:, 2] / traj[0, 2], label='mass')
        plt.xlabel("time (s)")
        plt.ylabel("mass (kg)")
        plt.show()


