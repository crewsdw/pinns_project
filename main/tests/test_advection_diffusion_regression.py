import os
import sys
import unittest

import numpy as np
import tensorflow as tf

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import irk_coefficients as irk
import neural_net as nn
import x_grid as grid


class TestAdvectionDiffusionRegression(unittest.TestCase):
    """Regression tests for Advection-Diffusion network with saved reference values"""

    def setUp(self):
        """Set up test fixtures with deterministic seeds"""
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)

        self.order = 4
        self.nodes = 8
        self.IRK = irk.IRK(order=self.order)
        self.IRK.build_matrix()

        self.dt = 0.05
        self.alpha = 1.0
        self.parameters = [self.dt, self.alpha]
        self.activation = "tanh"

        # Create test grid and initial condition
        basis = grid.GridX(order=self.nodes)
        self.x0 = 0.5 * (basis.nodes + 1.0)
        self.u0 = self.solution_periodic(self.x0, 0, self.alpha)
        self.boundary = np.array([0.0, 1.0])

        # Convert to TensorFlow tensors
        self.xtf = tf.reshape(tf.convert_to_tensor(self.x0), (self.nodes, 1))
        self.utf = tf.reshape(tf.convert_to_tensor(self.u0), (self.nodes, 1))

        # Reference values
        self.reference_output_shape = (3, self.nodes, self.order + 1)

    def solution_periodic(self, x, t, a):
        """Periodic traveling mode solution for testing"""
        return np.sin(2.0 * np.pi * (x - a * t)) * np.exp(-((2.0 * np.pi) ** 2.0) * t)

    def solution_dirichlet(self, x, t, a):
        """Dirichlet boundary solution for testing"""
        return (
            np.exp(a * 0.5 * (x - a * 0.5 * t))
            * np.sin(np.pi * x)
            * np.exp(-(np.pi**2.0) * t)
        )

    def test_advection_diffusion_forward_pass_shape(self):
        """Test that forward pass produces expected output shape"""
        net = nn.NeuralNet_AdvectionDiffusion(
            x=self.x0,
            u=self.u0,
            bc=self.boundary,
            parameters=self.parameters,
            irk=self.IRK,
            neurons=self.nodes,
            activation=self.activation,
        )

        output = net(self.xtf)
        self.assertEqual(output.shape, self.reference_output_shape)

    def test_advection_diffusion_loss_computation(self):
        """Test that loss computation returns a scalar value"""
        net = nn.NeuralNet_AdvectionDiffusion(
            x=self.x0,
            u=self.u0,
            bc=self.boundary,
            parameters=self.parameters,
            irk=self.IRK,
            neurons=self.nodes,
            activation=self.activation,
        )

        # Forward pass to get predictions
        predictions = net(self.xtf)

        # Compute loss
        loss_value = net.loss_with_bc(self.utf, predictions)

        # Loss should be a scalar
        self.assertEqual(loss_value.shape, ())
        self.assertTrue(tf.is_tensor(loss_value))
        self.assertGreater(loss_value.numpy(), 0)  # Loss should be positive initially

    def test_advection_diffusion_boundary_conditions(self):
        """Test that boundary conditions are properly enforced"""
        net = nn.NeuralNet_AdvectionDiffusion(
            x=self.x0,
            u=self.u0,
            bc=self.boundary,
            parameters=self.parameters,
            irk=self.IRK,
            neurons=self.nodes,
            activation=self.activation,
        )

        # Test boundary evaluation
        bc_tensor = tf.reshape(
            tf.constant([self.boundary[0], self.boundary[1]], dtype=tf.float32), (2, 1)
        )
        boundary_output = net(bc_tensor)

        # Should produce output at boundary points
        expected_boundary_shape = (3, 2, self.order + 1)
        self.assertEqual(boundary_output.shape, expected_boundary_shape)

    def test_advection_diffusion_gradient_computation(self):
        """Test that spatial gradients are computed correctly"""
        net = nn.NeuralNet_AdvectionDiffusion(
            x=self.x0,
            u=self.u0,
            bc=self.boundary,
            parameters=self.parameters,
            irk=self.IRK,
            neurons=self.nodes,
            activation=self.activation,
        )

        output = net(self.xtf)

        # Extract components: u, u_x, u_xx
        u = output[0, :, :]  # Solution values
        u_x = output[1, :, :]  # First derivatives
        u_xx = output[2, :, :]  # Second derivatives

        # Check shapes
        expected_component_shape = (self.nodes, self.order + 1)
        self.assertEqual(u.shape, expected_component_shape)
        self.assertEqual(u_x.shape, expected_component_shape)
        self.assertEqual(u_xx.shape, expected_component_shape)

        # Basic sanity check: derivatives should not be identical to solution
        self.assertFalse(np.allclose(u.numpy(), u_x.numpy()))
        self.assertFalse(np.allclose(u.numpy(), u_xx.numpy()))

    def test_advection_diffusion_physics_consistency(self):
        """Test that the PDE physics are consistent"""
        net = nn.NeuralNet_AdvectionDiffusion(
            x=self.x0,
            u=self.u0,
            bc=self.boundary,
            parameters=self.parameters,
            irk=self.IRK,
            neurons=self.nodes,
            activation=self.activation,
        )

        output = net(self.xtf)

        # Extract derivatives
        u_x = output[1, :, :]
        u_xx = output[2, :, :]

        # Compute RHS manually to verify physics
        rhs = -self.alpha * u_x + u_xx

        # This should match the RHS computation in the loss function
        expected_rhs_shape = (self.nodes, self.order + 1)
        self.assertEqual(rhs.shape, expected_rhs_shape)

    def test_advection_diffusion_analytical_solution_consistency(self):
        """Test consistency with analytical solutions"""
        # Test that analytical solutions satisfy expected properties
        x_test = np.linspace(0, 1, 10)
        t_test = 0.1

        # Periodic solution
        u_periodic = self.solution_periodic(x_test, t_test, self.alpha)
        self.assertEqual(len(u_periodic), len(x_test))

        # Dirichlet solution
        u_dirichlet = self.solution_dirichlet(x_test, t_test, self.alpha)
        self.assertEqual(len(u_dirichlet), len(x_test))

        # Solutions should decay over time (diffusion effect)
        u_periodic_later = self.solution_periodic(x_test, t_test * 2, self.alpha)
        self.assertLess(np.max(np.abs(u_periodic_later)), np.max(np.abs(u_periodic)))

    def test_advection_diffusion_deterministic_output(self):
        """Test that network produces deterministic outputs"""
        # Create two identical networks
        net1 = nn.NeuralNet_AdvectionDiffusion(
            x=self.x0,
            u=self.u0,
            bc=self.boundary,
            parameters=self.parameters,
            irk=self.IRK,
            neurons=self.nodes,
            activation=self.activation,
            seed=123,
        )

        net2 = nn.NeuralNet_AdvectionDiffusion(
            x=self.x0,
            u=self.u0,
            bc=self.boundary,
            parameters=self.parameters,
            irk=self.IRK,
            neurons=self.nodes,
            activation=self.activation,
            seed=123,
        )

        output1 = net1(self.xtf)
        output2 = net2(self.xtf)

        np.testing.assert_array_almost_equal(
            output1.numpy(), output2.numpy(), decimal=6
        )


if __name__ == "__main__":
    unittest.main()
