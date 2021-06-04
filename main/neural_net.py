import tensorflow as tf
from tensorflow.keras import backend as K

class NeuralNet_AdvectionDiffusion(tf.keras.Model):
    def __init__(self, x, u, bc, parameters, irk, neurons, activation='tanh'):
        super(NeuralNet_AdvectionDiffusion, self).__init__()
        # Variables
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
            u_x.append(tf.gradients(u1[:, i], x)[0])
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
        # dirichlet BCs...
        boundary1 = 0.0 * K.mean(K.square(b1[0, :]))
        boundary2 = 0.0 * K.mean(K.square(b1[1, :]))

        boundary3 = K.sum(K.square(b1[0, :] - b1[1, :]))

        return mean_sqr_error + boundary1 + boundary2 + boundary3

    def get_config(self):
        pass


class NeuralNet_LorenzStepper(tf.keras.Model):
    def __init__(self, parameters, irk, neurons, activation='tanh'):
        super(NeuralNet_LorenzStepper, self).__init__()
        # problem parameters
        self.dt = parameters[0]
        # lorenz sys parameters: original lorenz and fetter choices
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
        return self.net(x)

    # Loss function
    @tf.autograph.experimental.do_not_convert
    def loss(self, u0_true, u1):
        # Compute Lorenz system RHS
        rhs = tf.convert_to_tensor([self.sigma * (u1[:, 1, :] - u1[:, 0, :]),
                                    u1[:, 0, :] * (self.rho - u1[:, 2, :]) - u1[:, 1, :],
                                    u1[:, 0, :] * u1[:, 1, :] - self.beta * u1[:, 2, :]])
        # Objective function: match initial condition
        u0 = u1 - self.dt * tf.transpose(tf.matmul(rhs, self.irk.rk_matrix_tf32), perm=(1, 0, 2))

        # Error
        error = u0_true[:, :, None] - u0
        sqr_error = K.square(error)
        mean_sqr_error = K.mean(sqr_error)

        return mean_sqr_error

    def get_config(self):
        pass
