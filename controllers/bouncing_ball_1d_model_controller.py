import numpy as np
import numpy.linalg
import math
import tensorflow as tf


class BouncingBall1DModelLearning(object):
    def __init__(self, sess, coeff_of_restitution, desired_height=7.):
        self.state_dim = 2
        self.output_dim = 2
        self.coeff_of_restitution = coeff_of_restitution
        self.desired_height = desired_height

        self.sess = sess
        self._build_network()


    def _build_network(self):
        self.state_ph = tf.placeholder(tf.float32, [None, self.state_dim], name='state_ph')
        net = tf.layers.dense(self.state_ph, 64, activation=tf.nn.relu)
        net = tf.layers.dense(net, 32, activation=tf.nn.relu)
        self.output = tf.layers.dense(net, self.output_dim, activation=None)


        self.output_ph = tf.placeholder(tf.float32, [None, self.output_dim], name='output_ph')
        self.loss = tf.reduce_mean(tf.losses.mean_squared_error(self.output_ph, self.output))

        self.optimizer = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

    def train(self, states, outputs):
        fd = {self.state_ph: states, self.output_ph: outputs}

        loss, _ = self.sess.run([self.loss, self.optimizer], feed_dict=fd)
        return loss


    def get_loss(self, states, outputs):
        fd = {self.state_ph: states, self.output_ph: outputs}

        loss = self.sess.run(self.loss, feed_dict=fd)
        return loss

    def infer(self, states):
        fd = {self.state_ph: states}
        output = self.sess.run(self.output, feed_dict=fd)
        return output


    def get_control(self, state, control):
        # state is dim 1 array
        next_state = self.infer(np.array([[ state[1], control[0] ]]))
        next_state = np.reshape(next_state, (self.output_dim,))
        print("Next state: " + str(next_state))

        grav = -9.81
        time_until_impact = next_state[0]
        ball_vel_at_impact = next_state[1]

        desired_vel_after_impact = np.sqrt(2*-grav*self.desired_height)
        desired_paddle_vel = (desired_vel_after_impact + ball_vel_at_impact) / (self.coeff_of_restitution + 1)

        # fit a quadratic path to the controller
        tau = time_until_impact
        coeffs = np.zeros(3) # from highest ordered coeff to lowest
        A = np.array([[tau**2, tau], [2*tau, 1.]])
        b = np.array([[0., desired_paddle_vel]]).T - np.array([[state[2], 0]]).T
        sol = np.linalg.solve(A, b).squeeze()
        coeffs[0] = sol[0]
        coeffs[1] = sol[1]
        coeffs[2] = state[2]

        return coeffs

    def save_model(self, filename='/tmp/model.ckpt'):
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, filename)
        print("Model saved in file: %s" % filename)

    def load_model(self, filename='/tmp/model.ckpt'):
        saver = tf.train.Saver()
        saver.restore(self.sess, filename)
        print("Model loaded from file: %s" % filename)



