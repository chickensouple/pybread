import numpy as np
import math
import pybread
from pybread.controllers.bouncing_ball_1d_controller import *
from pybread.controllers.bouncing_ball_1d_model_controller import *
from pybread.systems import BouncingBall1DModel
import matplotlib.pyplot as plt


def get_trajectory(model, controller, initial_state, T=2000, dt=0.01):

    state_hist = np.zeros((T, 3))
    control_hist = np.zeros((T, 1))

    collision_list = [] # list of [time, ball_vel, paddle_vel]

    state = initial_state
    for t in range(T):
        action = controller.get_control(state)
        new_state = model.get_next_state(state, action, dt)

        # if ball velocity is flipped and position is close to ground
        if abs(state[1] - new_state[1]) > abs(state[1]) and state[0] < 0.5:
            collision_list.append([t*dt, state[1], action[0]])

        state_hist[t, :] = state
        control_hist[t, :] = action
        state = new_state

    collision_list = np.array(collision_list)
    return state_hist, control_hist, collision_list


def gather_data(desired_height_list):
    restitution = 0.9

    model = BouncingBall1DModel(max_vel=float('inf'), drag_coeff=0., bounce_restitution=restitution)

    dt = 0.01
    T = 3000
    state_hist = np.zeros((T, 3))
    control_hist = np.zeros((T, 1))

    state = np.array([5., 0., -2])

    controller_list = [BouncingBall1DController(restitution, desired_height) for desired_height in desired_height_list]    

    train_state_list = []
    train_next_state_list = []
    for controller in controller_list:
        state_hist, control_hist, collision_list = get_trajectory(model, controller, state, T, dt)

        train_state = collision_list[:-1, :]
        train_next_state = collision_list[1:, :]

        train_state_list.append(train_state)
        train_next_state_list.append(train_next_state)

    train_state_list = np.concatenate(train_state_list, axis=0)
    train_next_state_list = np.concatenate(train_next_state_list, axis=0)
    x = train_state_list[:, 1:]
    y = np.stack([train_next_state_list[:, 0] - train_state_list[:, 0], train_next_state_list[:, 1]], axis=1)

    return train_state_list, train_next_state_list, x, y




if __name__ == '__main__':
    import argparse
    import pickle
    parser = argparse.ArgumentParser(description="Basic model learning")
    parser.add_argument("--data", dest="data", action="store_true")
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--test", dest="test", action="store_true")

    args = parser.parse_args()

    if args.data:
        desired_height_list = [4, 4.5, 5, 5.5, 6, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5]
        train_state_list, train_next_state_list, x, y = gather_data(desired_height_list)
        save_dict = {'state': train_state_list, 'next_state': train_next_state_list, 'x': x, 'y': y}
        pickle.dump(save_dict, open(pybread.__path__[0] + "/data/bouncing_ball_1d_train.p", "wb"))

        desired_height_list = [7, 7.5]
        train_state_list, train_next_state_list, x, y = gather_data(desired_height_list)
        save_dict = {'state': train_state_list, 'next_state': train_next_state_list, 'x': x, 'y': y}
        pickle.dump(save_dict, open(pybread.__path__[0] + "/data/bouncing_ball_1d_val.p", "wb"))
    if args.train:
        train_data = pickle.load(open(pybread.__path__[0] + "/data/bouncing_ball_1d_train.p", "rb"))
        val_data = pickle.load(open(pybread.__path__[0] + "/data/bouncing_ball_1d_val.p", "rb"))

        sess = tf.Session()
        controller = BouncingBall1DModelLearning(sess, 0.9, 7.)
        sess.run(tf.global_variables_initializer())

        batch_size = 64
        for i in range(1000):
            rand_idx = np.random.randint(len(train_data['x']), size=batch_size)
            x = train_data['x'][rand_idx, :]
            y = train_data['y'][rand_idx, :]

            loss = controller.train(x, y)
            val_loss = controller.get_loss(val_data['x'], val_data['y'])
            print("Iteration " + str(i) + ": train loss - " + str(loss) + "\tval loss - " + str(val_loss))
        controller.save_model()

    if args.test:
        restitution = 0.9
        desired_height = 7.

        sess = tf.Session()
        controller = BouncingBall1DModelLearning(sess, restitution, desired_height)
        sess.run(tf.global_variables_initializer())
        controller.load_model()

        model = BouncingBall1DModel(max_vel=float('inf'), drag_coeff=0., bounce_restitution=restitution)

        T = 1000
        dt = 0.01
        state = np.array([5., 0., -2])

        state_hist = np.zeros((T, 3))
        control_hist = np.zeros((T, 1))
        collision_list = []

        last_collision_t = 0.
        coeffs = np.zeros(3)
        for t in range(T):
            delta_t = t*dt - last_collision_t
            action = delta_t*delta_t*coeffs[0] + delta_t*coeffs[1] + coeffs[2]
            action = np.array([action])
            new_state = model.get_next_state(state, action, dt)

            # if ball velocity is flipped and position is close to ground
            if abs(state[1] - new_state[1]) > abs(state[1]) and state[0] < 0.5:
                collision_list.append([t*dt, state[1], action[0]])
                coeffs = controller.get_control(state, action)
                last_collision_t = t*dt

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

