import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


# from tensorflow.keras import layers


class NeuralNet_AdvectionDiffusion(tf.keras.Model):
    def __init__(self, x, u, bc, parameters, irk, neurons, activation='tanh'):
        super(NeuralNet_AdvectionDiffusion, self).__init__()
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
        u_x, u_xx = [], []
        for i in range(self.irk.order + 1):
            # print(u1)
            # print(inputs.shape)
            u_x.append(tf.gradients(u1[:, i], x)[0])
            # print(u_x)
            # quit()
            u_xx.append(tf.gradients(tf.gradients(u1[:, i], x)[0], x)[0])

        a = tf.stack(u_x)[:, :, 0]  # tf.reshape(tf.stack(u_x), (self.irk.order+1, self.neurons))
        b = tf.stack(u_xx)[:, :, 0]  # tf.reshape(tf.stack(u_xx), (self.irk.order+1, self.neurons))
        u1_x = tf.transpose(a, perm=(1, 0))
        u1_xx = tf.transpose(b, perm=(1, 0))

        return tf.stack([u1, u1_x, u1_xx])  # [u1, u1_x, u1_xx]

    # Loss function
    @tf.autograph.experimental.do_not_convert
    def loss_with_bc(self, u0_true, u1_prediction):
        # u1_prediction = self.call(prediction)
        u1 = u1_prediction[0, :, :]
        u1_x = u1_prediction[1, :, :]
        u1_xx = u1_prediction[2, :, :]
        rhs = -self.alpha * u1_x + u1_xx
        u0 = u1 - self.dt * tf.matmul(rhs, self.irk.rk_matrix_tf32)

        # Error
        error = u0_true - u0
        sqr_error = K.square(error)
        mean_sqr_error = K.sum(sqr_error)

        # L1 boundaries
        boundaries = self.call(self.bc)
        b1 = boundaries[0, :, :]

        boundary1 = 0.0 * K.mean(K.square(b1[0, :]))
        boundary2 = 0.0 * K.mean(K.square(b1[1, :]))

        boundary3 = K.sum(K.square(b1[0, :] - b1[1, :]))  # +

        return mean_sqr_error + boundary1 + boundary2 + boundary3

    def get_config(self):
        pass


class NeuralNet_LorenzStepper(tf.keras.Model):
    def __init__(self, parameters, irk, neurons, activation='tanh'):
        super(NeuralNet_LorenzStepper, self).__init__()
        # Variables
        # self.x = tf.convert_to_tensor(x)  # position
        # self.u = tf.convert_to_tensor(u)  # evolution variable initial condition
        # self.bc = bc  # boundary conditions

        # self.left = tf.constant([0])
        # self.right = tf.constant([-1])
        # self.bc = tf.reshape(tf.constant([bc[0], bc[1]], dtype=tf.float32), (2, 1))

        # problem parameters
        self.dt = parameters[0]
        # self.alpha = parameters[1]

        self.sigma = 10
        self.beta = 8 / 3
        self.rho = 28

        # runge-kutta coefficient matrix and weights
        self.irk = irk

        # net parameters
        self.neurons = neurons
        self.activation = activation

        self.net = tf.keras.Sequential([
            # tf.keras.layers.InputLayer(input_shape=(self.neurons, 1)),
            tf.keras.layers.Dense(self.neurons, activation=self.activation),
            tf.keras.layers.Dense(self.neurons, activation=self.activation),
            # tf.keras.layers.Dense(self.neurons, activation=self.activation),
            tf.keras.layers.Dense(self.neurons, activation=self.activation),
            tf.keras.layers.Dense(3 * (self.irk.order + 1), activation='linear'),
            tf.keras.layers.Reshape((3, self.irk.order + 1))  # self.activation),
        ])

    @tf.autograph.experimental.do_not_convert
    def call(self, x, training=None, mask=None):
        # output = self.net(inputs)
        # x1 = self.net(x[:, 0])
        # y1 = self.net(x[:, 1])
        # z1 = self.net(x[:, 2])
        # print(x)
        # print(x1)
        # print(y1)
        # print(z1)
        # print(x)
        u1 = self.net(x)
        # print(u1)
        # print(tf.reshape(u1, [11, 3, None]))
        # quit()

        # # Compute spatial gradients
        # u_x, u_xx = [], []
        # for i in range(self.irk.order + 1):
        #     # print(u1)
        #     # print(inputs.shape)
        #     u_x.append(tf.gradients(u1[:, i], x)[0])
        #     # print(u_x)
        #     # quit()
        #     u_xx.append(tf.gradients(tf.gradients(u1[:, i], x)[0], x)[0])

        # a = tf.stack(u_x)[:, :, 0]  # tf.reshape(tf.stack(u_x), (self.irk.order+1, self.neurons))
        # b = tf.stack(u_xx)[:, :, 0]  # tf.reshape(tf.stack(u_xx), (self.irk.order+1, self.neurons))
        # u1_x = tf.transpose(a, perm=(1, 0))
        # u1_xx = tf.transpose(b, perm=(1, 0))

        # for i in range(q.shape[0])])

        return u1  # x1, y1, z1  # tf.stack([u1, u1_x, u1_xx])  # [u1, u1_x, u1_xx]

    # Loss function
    @tf.autograph.experimental.do_not_convert
    def loss(self, u0_true, u1):
        # u1_prediction = self.call(prediction)
        # u1 = u1_prediction[0, :, :]
        # u1_x = u1_prediction[1, :, :]
        # u1_xx = u1_prediction[2, :, :]
        # print(u1)
        # rhs = tf.convert_to_tensor(np.array([self.sigma * (u1[1, :] - u1[0, :]),
        #                             u1[0, :] * (self.rho - u1[2, :]) - u1[1, :],
        #                             u1[0, :] * u1[1, :] - self.beta * u1[2, :]]))
        rhs = tf.convert_to_tensor([self.sigma * (u1[:, 1, :] - u1[:, 0, :]),
                                    u1[:, 0, :] * (self.rho - u1[:, 2, :]) - u1[:, 1, :],
                                    u1[:, 0, :] * u1[:, 1, :] - self.beta * u1[:, 2, :]])
        # print(rhs)
        # print(self.irk.rk_matrix_tf32.shape)
        # print(tf.matmul(rhs, self.irk.rk_matrix_tf32))
        # print(u1)
        # quit()
        # rhs = -self.alpha * u1_x + u1_xx
        u0 = u1 - self.dt * tf.transpose(tf.matmul(rhs, self.irk.rk_matrix_tf32), perm=(1, 0, 2))
        # print(u0)
        # print(u0_true)
        # quit()

        # Error
        error = u0_true[:, :, None] - u0
        sqr_error = K.square(error)
        mean_sqr_error = K.mean(sqr_error)

        # L1 boundaries
        # boundaries = self.call(self.bc)
        # b1 = boundaries[0, :, :]

        # boundary1 = 0.0 * K.mean(K.square(b1[0, :]))
        # boundary2 = 0.0 * K.mean(K.square(b1[1, :]))

        # boundary3 = K.sum(K.square(b1[0, :] - b1[1, :]))  # +

        return mean_sqr_error  # + boundary1 + boundary2 + boundary3

    def get_config(self):
        pass

# def call(self, x):
#     encoded = self.encoder(x)
#     decoded = self.decoder(encoded)
#     return decoded

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
