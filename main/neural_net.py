import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


# from tensorflow.keras import layers


class NeuralNet(tf.keras.Model):
    def __init__(self, x, u, bc, parameters, irk, neurons, activation='sigmoid'):
        super(NeuralNet, self).__init__()
        # Variables
        self.x = tf.convert_to_tensor(x)  # position
        self.u = tf.convert_to_tensor(u)  # evolution variable initial condition
        # self.bc = bc  # boundary conditions

        # self.left = tf.constant([0])
        # self.right = tf.constant([-1])
        self.bc = tf.reshape(tf.constant([bc[0], bc[1]], dtype=tf.float32), (2, 1))

        # problem parameters
        self.dt = parameters[0]
        self.alpha = parameters[1]

        # runge-kutta coefficient matrix and weights
        self.irk = irk

        # net parameters
        self.neurons = neurons
        self.activation = activation

        self.net = tf.keras.Sequential([
            # tf.keras.layers.InputLayer(input_shape=(self.neurons, 1)),
            tf.keras.layers.Dense(self.neurons, activation=self.activation),
            tf.keras.layers.Dense(self.neurons, activation=self.activation),
            tf.keras.layers.Dense(self.neurons, activation=self.activation),
            tf.keras.layers.Dense(self.neurons, activation=self.activation),
            tf.keras.layers.Dense(self.irk.order + 1, activation='linear')  # self.activation),
        ])

    @tf.autograph.experimental.do_not_convert
    def call(self, x, training=None, mask=None):
        # output = self.net(inputs)
        u1 = self.net(x)
        # Compute spatial gradients
        # print(u1)
        # quit()
        u_x, u_xx = [], []
        for i in range(self.irk.order + 1):
            # print(u1)
            # print(inputs.shape)
            u_x.append(tf.gradients(u1[:, i], x)[0])
            # print(u_x)
            # quit()
            u_xx.append(tf.gradients(tf.gradients(u1[:, i], x)[0], x)[0])
        # print('\nInput shape:')
        # print(inputs)
        # print('\nOutput shape:')
        # print(output)
        # print(u1)
        # print('\nGradient shapes:')
        # print(u1_x)
        # print(u1_xx)
        # quit()
        a = tf.stack(u_x)[:, :, 0]  # tf.reshape(tf.stack(u_x), (self.irk.order+1, self.neurons))
        b = tf.stack(u_xx)[:, :, 0]  # tf.reshape(tf.stack(u_xx), (self.irk.order+1, self.neurons))
        u1_x = tf.transpose(a, perm=(1, 0))
        u1_xx = tf.transpose(b, perm=(1, 0))
        # print(u1)
        # print(u1_x)
        # print(u1_xx)
        # quit()
        return tf.stack([u1, u1_x, u1_xx])  # [u1, u1_x, u1_xx]

    # Loss function
    @tf.autograph.experimental.do_not_convert
    def loss_with_bc(self, u0_true, u1_prediction):
        # u1_prediction = self.call(prediction)
        u1 = u1_prediction[0, :, :]
        u1_x = u1_prediction[1, :, :]
        u1_xx = u1_prediction[2, :, :]
        # print(u1_prediction)
        # print(u1)
        # print(u1_x)
        # print(u1_xx)
        # quit()

        rhs = -self.alpha * u1_x + u1_xx
        u0 = u1 - self.dt * tf.matmul(rhs, self.irk.rk_matrix_tf32)

        # Error
        error = u0_true - u0
        sqr_error = K.square(error)
        mean_sqr_error = K.sum(sqr_error)

        # L1 boundaries
        boundaries = self.call(self.bc)
        b1 = boundaries[0, :, :]
        # b1_x = boundaries[1, :, :]
        # b1_xx = boundaries[2, :, :]
        # print(b_1)
        # quit()
        # print(u0_true)
        # quit()

        boundary1 = 0.0 * K.mean(K.square(b1[0, :]))
        boundary2 = 0.0 * K.mean(K.square(b1[1, :]))

        boundary3 = K.sum(K.square(b1[0, :] - b1[1, :]))  # +
        # K.mean(K.square(b1_x[0, :] - b1_x[1, :])) +
        # K.mean(K.square(b1_xx[0, :] - b1_xx[1, :])))

        return mean_sqr_error + boundary1 + boundary2 + boundary3

        # # Compute RK stage vector, rhs = du/dt
        # rhs = 0.0  # -self.alpha * u1_x + u1_xx
        # u0 = u1  # - self.dt * tf.matmul(rhs, self.irk.rk_matrix_tf32)
        #
        # sse_n = 0
        # sse_bc = 0
        # for i in range(self.irk.order + 1):
        #     # sse_n += tf.reduce_sum(tf.square(u_true - u_prediction[:, i]))
        #     sse_n += tf.reduce_sum(tf.square(u0_true - u0[:, i]))
        #
        # # boundaries
        # boundaries = self.call(self.bc)
        # b_1 = boundaries[0, :, :]
        # b1_x = boundaries[1, :, :]
        # b1_xx = boundaries[2, :, :]
        #
        # rhs = -self.alpha * b1_x + b1_xx
        # b0 = b_1 - self.dt * tf.matmul(rhs, self.irk.rk_matrix_tf32)
        # # sse_bc += tf.reduce_sum(tf.abs(b0[0, :])) + tf.reduce_sum(tf.abs(b0[1, :]))
        # sse_bc = tf.reduce_sum(tf.square(b0[0, :] - b0[1, :]))
        # # sse_bc += u0[0, i] ** 2.0 + u0[-1, i] ** 2.0  # + tf.abs(u0[0, i] + u0[-1, i])
        # # sse_bc = tf.abs(self.call(inputs=self.bc[0])) + tf.abs(self.call(inputs=self.bc[1]))
        #
        # return sse_n + sse_bc

    def get_config(self):
        pass

    # def call(self, x):
    #     encoded = self.encoder(x)
    #     decoded = self.decoder(encoded)
    #     return decoded
