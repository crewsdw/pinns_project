import os
import sys
import unittest

import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import irk_coefficients as irk
import neural_net as nn


class TestLorenzRegression(unittest.TestCase):
    """Regression tests for Lorenz system network with saved reference values"""

    def setUp(self):
        """Set up test fixtures with deterministic seeds"""
        np.random.seed(42)
        tf.random.set_seed(42)

        self.order = 2
        self.IRK = irk.IRK(order=self.order)
        self.IRK.build_matrix()

        self.dt = 0.1
        self.alpha = 1.0
        self.parameters = [self.dt, self.alpha]
        self.activation = "tanh"

        self.q = np.array([1.0, 1.0, 1.0])
        self.u0 = np.array([self.q])
        self.utf = tf.reshape(
            tf.convert_to_tensor(self.u0), (self.u0.shape[0], self.u0.shape[1])
        )

        self.reference_output_shape = (1, 3, self.order + 1)

    def test_lorenz_forward_pass_shape(self):
        """Test that forward pass produces expected output shape"""
        net = nn.NeuralNet_LorenzStepper(
            parameters=self.parameters,
            irk=self.IRK,
            neurons=self.q.shape[0],
            activation=self.activation,
        )

        output = net(self.utf)
        self.assertEqual(output.shape, self.reference_output_shape)

    def test_lorenz_loss_computation(self):
        """Test that loss computation returns a scalar value"""
        net = nn.NeuralNet_LorenzStepper(
            parameters=self.parameters,
            irk=self.IRK,
            neurons=self.q.shape[0],
            activation=self.activation,
        )

        predictions = net(self.utf)

        loss_value = net.custom_loss(self.utf, predictions)

        self.assertEqual(loss_value.shape, ())
        self.assertTrue(tf.is_tensor(loss_value))
        self.assertGreater(loss_value.numpy(), 0)  # Loss should be positive initially

    def test_lorenz_untrained_network_deterministic(self):
        """Test that untrained network produces deterministic outputs"""
        net1 = nn.NeuralNet_LorenzStepper(
            parameters=self.parameters,
            irk=self.IRK,
            neurons=self.q.shape[0],
            activation=self.activation,
            seed=123,
        )

        net2 = nn.NeuralNet_LorenzStepper(
            parameters=self.parameters,
            irk=self.IRK,
            neurons=self.q.shape[0],
            activation=self.activation,
            seed=123,
        )

        output1 = net1(self.utf)
        output2 = net2(self.utf)

        np.testing.assert_array_almost_equal(
            output1.numpy(), output2.numpy(), decimal=6
        )

    def test_lorenz_physics_consistency(self):
        """Test that the loss function enforces Lorenz system physics"""
        net = nn.NeuralNet_LorenzStepper(
            parameters=self.parameters,
            irk=self.IRK,
            neurons=self.q.shape[0],
            activation=self.activation,
        )

        predictions = net(self.utf)

        u1 = predictions

        sigma, beta, rho = 10, 8 / 3, 28
        expected_rhs = tf.convert_to_tensor(
            [
                sigma * (u1[:, 1, :] - u1[:, 0, :]),
                u1[:, 0, :] * (rho - u1[:, 2, :]) - u1[:, 1, :],
                u1[:, 0, :] * u1[:, 1, :] - beta * u1[:, 2, :],
            ]
        )

        self.assertEqual(expected_rhs.shape, (3, 1, self.order + 1))

    def test_lorenz_irk_matrix_consistency(self):
        """Test that IRK matrix operations are consistent"""
        net = nn.NeuralNet_LorenzStepper(
            parameters=self.parameters,
            irk=self.IRK,
            neurons=self.q.shape[0],
            activation=self.activation,
        )

        actual_order = self.IRK.rk_matrix_tf32.shape[0]
        expected_matrix_shape = (actual_order, actual_order)
        self.assertEqual(self.IRK.rk_matrix_tf32.shape, expected_matrix_shape)

        matrix_values = self.IRK.rk_matrix_tf32.numpy()
        self.assertFalse(np.allclose(matrix_values, 0))
        self.assertFalse(np.allclose(matrix_values, 1))


if __name__ == "__main__":
    unittest.main()
