from dm_control import mjcf
import dm_control
import dm_control.suite
from dm_control.suite import common
import collections
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import cvxpy


class EmptyTask(dm_control.suite.base.Task):
    def __init__(self):
        super(EmptyTask, self).__init__(random=False)

    def initialize_episode(self, physics):
        # physics.named.data.qpos[['left_arm/shoulder', 'left_arm/elbow', 'right_arm/shoulder', 'right_arm/elbow']] = np.random.random(4) 
        pass

    def get_observation(self, physics):
        return None

    def get_reward(self, physics):
        return 0




class EnergyShapingPolicy(object):
    def __init__(self, env, height, loc):
        self.env = env
        self.height = height
        self.loc = loc
        self.hit_height = 0.5
        self.grav = -9.81
        self.alpha = 0.8 # coef of restitution

        self.vel_controller = PaddleVelController(env)

    def get_state(self):
        ball_pos = self.env._physics.named.data.qpos["ball"][:3]
        ball_vel = self.env._physics.named.data.qvel["ball"][:3]
        ball_acc = self.env._physics.named.data.qacc["ball"][:3]
        paddle_pos = self.env._physics.named.data.qpos[["paddle_x", "paddle_y", "paddle_z"]]
        paddle_vel = self.env._physics.named.data.qvel[["paddle_x", "paddle_y", "paddle_z"]]
        paddle_acc = self.env._physics.named.data.qacc[["paddle_x", "paddle_y", "paddle_z"]]
        paddle_orientation = self.env._physics.named.data.qpos[["paddle_roll", "paddle_pitch"]]
        # paddle_state = self.env._physics.named.data.qpos[["paddle_x", "paddle_y", "paddle_z", "paddle_roll", "paddle_pitch"]]

        state = np.concatenate([ball_pos, ball_vel, ball_acc, paddle_pos, paddle_vel, paddle_acc, paddle_orientation])
        # state is [ball_pos_x, ball_pos_y, ball_pos_z, 
        #           ball_vel_x, ball_vel_y, ball_vel_z, 
        #           paddle_pos_x, paddle_pos_y, paddle_pos_z, 
        #           paddle_roll, paddle_pitch]
        return state

    def get_action(self, timestep):
        # cur_state = self.env.
        curr_state = self.get_state()
        # print(curr_state)

        ball_pos = curr_state[:3]
        ball_vel = curr_state[3:6]
        ball_acc = curr_state[6:9]
        paddle_pos = curr_state[9:12]
        paddle_vel = curr_state[12:15]
        paddle_acc = curr_state[15:18]
        paddle_orientation = curr_state[18:]

        # get impact location and ball velocity at z = HIT HEIGHT plane
        delta_z = ball_pos[2] - self.hit_height

        # if delta_z < 0:
        #     delta_z = 0
        # print("Delta_z: " + str(delta_z))

        if (ball_vel[2]**2 - 2*self.grav*delta_z < 0):
            return np.zeros(5)

        time_until_impact = (-ball_vel[2] - np.sqrt(ball_vel[2]**2 - 2*self.grav*delta_z)) / self.grav
        # time_until_impact = 0.2
        ball_vel_at_impact = np.copy(ball_vel)
        ball_vel_at_impact[2] += self.grav*time_until_impact

        impact_loc = np.zeros(3)
        impact_loc[2] = self.hit_height
        impact_loc[0:2] = ball_vel[0:2] * time_until_impact + ball_pos[0:2]

        # compute desired velocity after hitting ball
        desired_ball_vel = np.zeros(3)
        desired_ball_vel[2] = np.sqrt(2*(self.height - self.hit_height) * -self.grav)
        time_to_crest = np.sqrt(2*(self.height - self.hit_height) / -self.grav)
        desired_ball_vel[0:2] = (self.loc - impact_loc[0:2]) / time_to_crest

        # compute desired paddle orientation and velocity
        p = np.array([0, 0, 1])
        vpaddle = self.solve_vpaddle_given_paddle_orientation(desired_ball_vel, ball_vel_at_impact, p, self.alpha)

        # compute path for paddle
        coeffs = []
        for i in range(3):
            coeffs.append(self.get_polynomial_path(paddle_pos[i], impact_loc[i], vpaddle[i], time_until_impact))
        # TODO: give path for paddle orientation

        curr_d_vel = np.array([coeffs[0][1], coeffs[1][1], coeffs[2][1]])
        action = self.vel_controller.get_action(curr_d_vel, 0., 0.)
        return action 
        # return np.array([0, 0, 0.1, 0., 0])


    def solve_vpaddle_given_paddle_orientation(self, vbounce, vball, p, alpha):
        # p is paddle orientation
        b = vbounce - (vball - 2*p*np.dot(p, vball))*alpha
        A = np.dot(np.array([p]).T, np.array([p])) + np.eye(3) * (1 - alpha)
        vpaddle = np.linalg.solve(A, b)
        return vpaddle

    def get_polynomial_path(self, start_pos, end_pos, end_vel, time):
        A = np.zeros((2, 2))
        A[0, 0] = time**2
        A[0, 1] = time
        A[1, 0] = 2*time
        A[1, 1] = 1
        b = np.array([end_pos - start_pos, end_vel])

        sol = np.linalg.solve(A, b)
        coeffs = np.zeros(3)
        coeffs[2] = start_pos
        coeffs[0] = sol[0]
        coeffs[1] = sol[1]
        return coeffs

class PID(object):
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0

    def get_action(self, delta, deriv, dt, debug=False):
        p = self.kp * delta
        i = self.ki * self.integral
        d = self.kd * deriv
        action = p + i + d

        self.integral += delta * dt

        if debug:
            data_dict = {"p": p,
                         "i": i,
                         "d": d,
                         "delta" : delta,
                         "action" : action,
                         "integral": self.integral}
            return action, data_dict


        return action


class PaddleVelController(object):
    def __init__(self, env):
        self.env = env
        self.pos_controller = [PID(20., 350, 0), PID(20., 350, 0), PID(20., 350, 0)]
        self.roll_controller = PID(0.01, 0.1, 0)
        self.pitch_controller = PID(0.01, 0.1, 0)

        self.paddle_mass = 0.1
        # self.pos_controller[2].integral = self.paddle_mass * 9.81 / self.pos_controller[2].ki

    def get_state(self):
        ball_pos = self.env._physics.named.data.qpos["ball"][:3]
        ball_vel = self.env._physics.named.data.qvel["ball"][:3]
        ball_acc = self.env._physics.named.data.qacc["ball"][:3]
        paddle_pos = self.env._physics.named.data.qpos[["paddle_x", "paddle_y", "paddle_z"]]
        paddle_vel = self.env._physics.named.data.qvel[["paddle_x", "paddle_y", "paddle_z"]]
        paddle_acc = self.env._physics.named.data.qacc[["paddle_x", "paddle_y", "paddle_z"]]
        paddle_orientation = self.env._physics.named.data.qpos[["paddle_roll", "paddle_pitch"]]
        # paddle_state = self.env._physics.named.data.qpos[["paddle_x", "paddle_y", "paddle_z", "paddle_roll", "paddle_pitch"]]

        state = np.concatenate([ball_pos, ball_vel, ball_acc, paddle_pos, paddle_vel, paddle_acc, paddle_orientation])
        # state is [ball_pos_x, ball_pos_y, ball_pos_z, 
        #           ball_vel_x, ball_vel_y, ball_vel_z, 
        #           paddle_pos_x, paddle_pos_y, paddle_pos_z, 
        #           paddle_roll, paddle_pitch]
        return state

    def get_action(self, desired_vel, desired_roll_vel, desired_pitch_vel, debug=False):
        paddle_acc = self.env._physics.named.data.qacc[["paddle_x", "paddle_y", "paddle_z"]]
        paddle_vel = self.env._physics.named.data.qvel[["paddle_x", "paddle_y", "paddle_z"]]

        paddle_o_acc = self.env._physics.named.data.qacc[["paddle_roll", "paddle_pitch"]]
        paddle_o_vel = self.env._physics.named.data.qvel[["paddle_roll", "paddle_pitch"]]

        dt = self.env._physics.timestep()
        action = np.zeros(5)

        if debug:
            data_dict = dict()

        for i in range(3):
            delta = desired_vel[i] - paddle_vel[i]
            if debug:
                action[i], data = self.pos_controller[i].get_action(delta, paddle_acc[i], dt, debug=True)
                data_dict["pos_" + str(i)] = data
            else:
                action[i] = self.pos_controller[i].get_action(delta, paddle_acc[i], dt)

        if debug:
            action[3], data = self.roll_controller.get_action(desired_roll_vel - paddle_o_vel[0], paddle_o_acc[0], dt, debug=True)
            data_dict["roll"] = data
            action[4], data = self.roll_controller.get_action(desired_roll_vel - paddle_o_vel[1], paddle_o_acc[1], dt, debug=True)
            data_dict["pitch"] = data
        else:
            action[3] = self.roll_controller.get_action(desired_roll_vel - paddle_o_vel[0], paddle_o_acc[0], dt)
            action[4] = self.roll_controller.get_action(desired_roll_vel - paddle_o_vel[1], paddle_o_acc[1], dt)

        if debug:
            return action, data_dict


        # print("paddle vel: " + str(paddle_vel) + "\t" + str(self.pos_controller[2].integral) + "\t" + str(action[2]))
        # print("paddle_roll: " + str(paddle_o_vel[0]))
        return action



if __name__ == '__main__':
    # TODO: test restitution
    import pybread
    import os
    from dm_control import viewer
    from dm_control.utils import io as resources

    def get_model_and_assets():
        filename = os.path.join(os.path.dirname(pybread.__file__), "systems", "ball_paddle.xml")
        """Returns a tuple containing the model XML string and a dict of assets."""
        return resources.GetResource(filename), common.ASSETS

    physics = mjcf.Physics.from_xml_string(*get_model_and_assets())
    empty_task = EmptyTask()

    env = dm_control.rl.control.Environment(physics=physics, task=empty_task, time_limit=50)


    policy = EnergyShapingPolicy(env, 1., np.array([0., 0]))
    policy_func = partial(EnergyShapingPolicy.get_action, policy)
    print(policy.get_state())

    # desired_vel = 0.1
    # policy = PaddleVelController(env)
    # policy_func = lambda time_step: policy.get_action(
    #     desired_vel=np.array([0, 0, desired_vel]),
    #     desired_roll_vel=0.1,
    #     desired_pitch_vel=0)
    # policy_func_debug = lambda time_step: policy.get_action(
    #     desired_vel=np.array([0, 0, desired_vel]),
    #     desired_roll_vel=0.1,
    #     desired_pitch_vel=0,
    #     debug=True)


    # plt_img = None
    # fig = plt.figure(1)
    # fig.show()

    # # val_type = "pos_2"
    # val_type = "roll"
    # vel_list = []
    # p_list = []
    # i_list = []
    # d_list = []
    # delta_list = []
    # integral_list = []
    # action_list = []

    # num_steps = 1000
    # dt = env._physics.timestep()
    # for _ in range(num_steps):


    #     action, data_dict = policy_func_debug(None)
    #     env._physics.set_control(action)
    #     env._physics.step()
    #     # vel_list.append(policy.get_state()[14])
    #     vel_list.append(env._physics.named.data.qvel["paddle_roll"])
    #     p_list.append(data_dict[val_type]['p'])
    #     i_list.append(data_dict[val_type]['i'])
    #     d_list.append(data_dict[val_type]['d'])
    #     delta_list.append(data_dict[val_type]['delta'])
    #     integral_list.append(data_dict[val_type]['integral'])
    #     action_list.append(data_dict[val_type]['action'])
    #     # img = env._physics.render()
    #     # if plt_img is None:
    #     #     plt_img = plt.imshow(img)
    #     # else:
    #     #     plt_img.set_data(img)
    #     # fig.canvas.draw()

    # times = [dt*i for i in range(num_steps)]
    # plt.figure(1)
    # plt.plot(times, vel_list, label="actual")
    # plt.plot(times, [desired_vel for _ in range(num_steps)], label="desired")
    # plt.legend()

    # plt.figure(2)
    # plt.plot(times, p_list, label="p")
    # plt.plot(times, i_list, label="i")
    # plt.plot(times, d_list, label="d")
    # plt.plot(times, action_list, label="action")
    # plt.legend()


    # plt.show()



    viewer.launch(env, policy=policy_func)
