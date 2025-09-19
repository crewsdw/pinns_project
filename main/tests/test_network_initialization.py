import os
import sys
import unittest

import numpy as np
import tensorflow as tf

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import irk_coefficients as irk
import neural_net as nn


class TestNetworkInitialization(unittest.TestCase):
    """Test minimal network initialization without training"""

    def setUp(self):
        """Set up test fixtures"""
        self.order = 4
        self.nodes = 8
        self.IRK = irk.IRK(order=self.order)
        self.IRK.build_matrix()

        # Parameters for both networks
        self.dt = 0.1
        self.alpha = 1.0
        self.parameters = [self.dt, self.alpha]
        self.activation = "tanh"

    def test_lorenz_stepper_initialization(self):
        """Test LorenzStepper network can be initialized"""
        neurons = 3
        net = nn.NeuralNet_LorenzStepper(
            parameters=self.parameters,
            irk=self.IRK,
            neurons=neurons,
            activation=self.activation,
        )

        # Check network parameters
        self.assertEqual(net.dt, self.dt)
        self.assertEqual(net.sigma, 10)
        self.assertEqual(net.beta, 8 / 3)
        self.assertEqual(net.rho, 28)
        self.assertEqual(net.neurons, neurons)
        self.assertEqual(net.activation, self.activation)

        # Test forward pass with dummy input
        u0 = np.array([[1.0, 2.0, 3.0]])
        utf = tf.reshape(tf.convert_to_tensor(u0), (u0.shape[0], u0.shape[1]))

        # Should not raise an error
        output = net(utf)
        expected_shape = (1, 3, self.order + 1)
        self.assertEqual(output.shape, expected_shape)

    def test_advection_diffusion_initialization(self):
        """Test AdvectionDiffusion network can be initialized"""
        x0 = np.linspace(0, 1, self.nodes)
        u0 = np.sin(np.pi * x0)
        boundary = np.array([0.0, 1.0])

        net = nn.NeuralNet_AdvectionDiffusion(
            x=x0,
            u=u0,
            bc=boundary,
            parameters=self.parameters,
            irk=self.IRK,
            neurons=self.nodes,
            activation=self.activation,
        )

        # Check network parameters
        self.assertEqual(net.dt, self.dt)
        self.assertEqual(net.alpha, self.alpha)
        self.assertEqual(net.neurons, self.nodes)
        self.assertEqual(net.activation, self.activation)

        # Test forward pass
        xtf = tf.reshape(tf.convert_to_tensor(x0), (self.nodes, 1))

        # Should not raise an error
        output = net(xtf)
        expected_shape = (3, self.nodes, self.order + 1)
        self.assertEqual(output.shape, expected_shape)

    def test_custom_loss_functions_exist(self):
        """Test that custom loss functions are callable"""
        # Lorenz stepper
        net_lorenz = nn.NeuralNet_LorenzStepper(
            parameters=self.parameters,
            irk=self.IRK,
            neurons=3,
            activation=self.activation,
        )
        self.assertTrue(callable(net_lorenz.custom_loss))

        # Advection diffusion
        x0 = np.linspace(0, 1, self.nodes)
        u0 = np.sin(np.pi * x0)
        boundary = np.array([0.0, 1.0])

        net_ad = nn.NeuralNet_AdvectionDiffusion(
            x=x0,
            u=u0,
            bc=boundary,
            parameters=self.parameters,
            irk=self.IRK,
            neurons=self.nodes,
            activation=self.activation,
        )
        self.assertTrue(callable(net_ad.loss_with_bc))


if __name__ == "__main__":
    unittest.main()
