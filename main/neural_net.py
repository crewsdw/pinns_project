import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


# Set deterministic behavior
def set_deterministic_seeds(seed=42):
    """Set all random seeds for deterministic behavior"""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # For TensorFlow 2.x deterministic operations
    try:
        tf.config.experimental.enable_op_determinism()
    except AttributeError:
        # Fallback for older TensorFlow versions
        import os

        os.environ["TF_DETERMINISTIC_OPS"] = "1"


# Set global deterministic behavior when module is imported
set_deterministic_seeds(42)


class NeuralNet_AdvectionDiffusion(tf.keras.Model):
    def __init__(self, x, u, bc, parameters, irk, neurons, activation="tanh", seed=42):
        super(NeuralNet_AdvectionDiffusion, self).__init__()
        # Set deterministic seeds for reproducible initialization
        set_deterministic_seeds(seed)
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

        # Use deterministic weight initialization
        initializer = tf.keras.initializers.GlorotUniform(seed=seed)

        self.net = tf.keras.Sequential(
            [
                # tf.keras.layers.InputLayer(input_shape=(self.neurons, 1)),
                tf.keras.layers.Dense(
                    self.neurons,
                    activation=self.activation,
                    kernel_initializer=initializer,
                    bias_initializer="zeros",
                ),
                tf.keras.layers.Dense(
                    self.neurons,
                    activation=self.activation,
                    kernel_initializer=initializer,
                    bias_initializer="zeros",
                ),
                tf.keras.layers.Dense(
                    self.neurons,
                    activation=self.activation,
                    kernel_initializer=initializer,
                    bias_initializer="zeros",
                ),
                tf.keras.layers.Dense(
                    self.neurons,
                    activation=self.activation,
                    kernel_initializer=initializer,
                    bias_initializer="zeros",
                ),
                tf.keras.layers.Dense(
                    self.irk.order + 1,
                    activation="linear",
                    kernel_initializer=initializer,
                    bias_initializer="zeros",
                ),  # self.activation),
            ]
        )

    @tf.autograph.experimental.do_not_convert
    def call(self, x, training=None, mask=None):
        # Compute spatial gradients using GradientTape
        u_x, u_xx = [], []

        for i in range(self.irk.order + 1):
            with tf.GradientTape() as tape2:
                tape2.watch(x)
                with tf.GradientTape() as tape1:
                    tape1.watch(x)
                    u1 = self.net(x)
                    u_i = u1[:, i]
                u_x_i = tape1.gradient(u_i, x)
            u_xx_i = tape2.gradient(u_x_i, x)

            u_x.append(u_x_i)
            u_xx.append(u_xx_i)

        # Get the full output for return
        u1 = self.net(x)

        a = tf.stack(u_x)[
            :, :, 0
        ]  # tf.reshape(tf.stack(u_x), (self.irk.order+1, self.neurons))
        b = tf.stack(u_xx)[
            :, :, 0
        ]  # tf.reshape(tf.stack(u_xx), (self.irk.order+1, self.neurons))
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
        error = tf.cast(u0_true, tf.float32) - u0
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
        config = super().get_config()
        config.update({
            'dt': self.dt,
            'alpha': self.alpha,
            'neurons': self.neurons,
            'activation': self.activation,
        })
        return config


class NeuralNet_LorenzStepper(tf.keras.Model):
    def __init__(self, parameters, irk, neurons, activation="tanh", seed=42):
        super(NeuralNet_LorenzStepper, self).__init__()
        # Set deterministic seeds for reproducible initialization
        set_deterministic_seeds(seed)
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

        # Use deterministic weight initialization
        initializer = tf.keras.initializers.GlorotUniform(seed=seed)

        self.net = tf.keras.Sequential(
            [
                # tf.keras.layers.InputLayer(input_shape=(self.neurons, 1)),
                tf.keras.layers.Dense(
                    self.neurons,
                    activation=self.activation,
                    kernel_initializer=initializer,
                    bias_initializer="zeros",
                ),
                tf.keras.layers.Dense(
                    self.neurons,
                    activation=self.activation,
                    kernel_initializer=initializer,
                    bias_initializer="zeros",
                ),
                tf.keras.layers.Dense(
                    self.neurons,
                    activation=self.activation,
                    kernel_initializer=initializer,
                    bias_initializer="zeros",
                ),
                tf.keras.layers.Dense(
                    3 * (self.irk.order + 1),
                    activation="linear",
                    kernel_initializer=initializer,
                    bias_initializer="zeros",
                ),
                tf.keras.layers.Reshape((3, self.irk.order + 1)),  # self.activation),
            ]
        )

    @tf.autograph.experimental.do_not_convert
    def call(self, x, training=None, mask=None):
        return self.net(x)

    # Loss function
    @tf.autograph.experimental.do_not_convert
    def custom_loss(self, u0_true, u1):
        # Compute Lorenz system RHS
        rhs = tf.convert_to_tensor(
            [
                self.sigma * (u1[:, 1, :] - u1[:, 0, :]),
                u1[:, 0, :] * (self.rho - u1[:, 2, :]) - u1[:, 1, :],
                u1[:, 0, :] * u1[:, 1, :] - self.beta * u1[:, 2, :],
            ]
        )
        # Objective function: match initial condition
        u0 = u1 - self.dt * tf.transpose(
            tf.matmul(rhs, self.irk.rk_matrix_tf32), perm=(1, 0, 2)
        )

        # Error
        error = tf.cast(u0_true[:, :, None], tf.float32) - u0
        sqr_error = K.square(error)
        mean_sqr_error = K.mean(sqr_error)

        return mean_sqr_error

    def get_config(self):
        config = super().get_config()
        config.update({
            'dt': self.dt,
            'sigma': self.sigma,
            'beta': self.beta,
            'rho': self.rho,
            'neurons': self.neurons,
            'activation': self.activation,
        })
        return config
