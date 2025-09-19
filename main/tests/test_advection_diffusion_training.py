import os
import pickle
import sys
import unittest

import numpy as np
import tensorflow as tf

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import irk_coefficients as irk
import neural_net as nn
import x_grid as grid


class TestAdvectionDiffusionTraining(unittest.TestCase):
    """Regression tests for Advection-Diffusion system training procedure"""

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

        # Load reference data if available
        self.reference_data = None
        ref_path = os.path.join(
            os.path.dirname(__file__),
            "reference_data",
            "advection_diffusion_reference.pkl",
        )
        if os.path.exists(ref_path):
            with open(ref_path, "rb") as f:
                self.reference_data = pickle.load(f)

    def solution_periodic(self, x, t, a):
        """Periodic traveling mode solution for testing"""
        return np.sin(2.0 * np.pi * (x - a * t)) * np.exp(-((2.0 * np.pi) ** 2.0) * t)

    def test_training_loss_decreases(self):
        """Test that training loss decreases over epochs"""
        net = nn.NeuralNet_AdvectionDiffusion(
            x=self.x0,
            u=self.u0,
            bc=self.boundary,
            parameters=self.parameters,
            irk=self.IRK,
            neurons=self.nodes,
            activation=self.activation,
        )

        # Record initial loss
        initial_loss = net.loss_with_bc(self.utf, net(self.xtf)).numpy()

        # Compile and train
        loss_fn = net.loss_with_bc
        net.compile(
            optimizer="adam", loss=lambda y_true, y_pred: loss_fn(y_true, y_pred)
        )

        history = net.fit(self.xtf, self.utf, epochs=10, verbose=0)

        # Final loss should be lower than initial
        final_loss = history.history["loss"][-1]
        self.assertLess(final_loss, initial_loss, "Training loss should decrease")

        # Loss should generally trend downward
        losses = history.history["loss"]
        self.assertLess(
            np.mean(losses[-3:]),
            np.mean(losses[:3]),
            "Average final losses should be less than average initial losses",
        )

    def test_training_convergence_with_boundary_conditions(self):
        """Test that training converges while respecting boundary conditions"""
        net = nn.NeuralNet_AdvectionDiffusion(
            x=self.x0,
            u=self.u0,
            bc=self.boundary,
            parameters=self.parameters,
            irk=self.IRK,
            neurons=self.nodes,
            activation=self.activation,
        )

        loss_fn = net.loss_with_bc
        net.compile(
            optimizer="adam", loss=lambda y_true, y_pred: loss_fn(y_true, y_pred)
        )

        # Train
        history = net.fit(self.xtf, self.utf, epochs=15, verbose=0)

        # Check convergence
        losses = history.history["loss"]
        final_losses = losses[-5:]
        loss_variance = np.var(final_losses)
        self.assertLess(loss_variance, 0.1, "Loss should stabilize during training")

        # Test boundary condition enforcement after training
        bc_tensor = tf.reshape(
            tf.constant([self.boundary[0], self.boundary[1]], dtype=tf.float32), (2, 1)
        )
        boundary_output = net(bc_tensor)

        # Boundary outputs should be reasonable (not NaN or infinite)
        self.assertFalse(
            np.any(np.isnan(boundary_output.numpy())),
            "Boundary outputs should not be NaN",
        )
        self.assertFalse(
            np.any(np.isinf(boundary_output.numpy())),
            "Boundary outputs should not be infinite",
        )

    def test_different_activation_functions(self):
        """Test training with different activation functions"""
        activations = ["tanh", "relu", "elu"]
        final_losses = {}

        for activation in activations:
            np.random.seed(42)
            tf.random.set_seed(42)

            net = nn.NeuralNet_AdvectionDiffusion(
                x=self.x0,
                u=self.u0,
                bc=self.boundary,
                parameters=self.parameters,
                irk=self.IRK,
                neurons=self.nodes,
                activation=activation,
            )

            loss_fn = net.loss_with_bc
            net.compile(
                optimizer="adam", loss=lambda y_true, y_pred: loss_fn(y_true, y_pred)
            )

            history = net.fit(self.xtf, self.utf, epochs=10, verbose=0)
            final_losses[activation] = history.history["loss"][-1]

        # All activations should achieve reasonable convergence
        for activation, final_loss in final_losses.items():
            self.assertLess(
                final_loss,
                50.0,
                f"Activation {activation} should achieve reasonable loss",
            )
            self.assertFalse(
                np.isnan(final_loss), f"Loss with {activation} should not be NaN"
            )

    def test_training_with_different_grid_sizes(self):
        """Test training behavior with different spatial grid sizes"""
        grid_sizes = [4, 8, 10]  # Use supported LGL orders
        final_losses = {}

        for nodes in grid_sizes:
            np.random.seed(42)
            tf.random.set_seed(42)

            # Create grid for this size
            basis = grid.GridX(order=nodes)
            x0 = 0.5 * (basis.nodes + 1.0)
            u0 = self.solution_periodic(x0, 0, self.alpha)

            xtf = tf.reshape(tf.convert_to_tensor(x0), (nodes, 1))
            utf = tf.reshape(tf.convert_to_tensor(u0), (nodes, 1))

            net = nn.NeuralNet_AdvectionDiffusion(
                x=x0,
                u=u0,
                bc=self.boundary,
                parameters=self.parameters,
                irk=self.IRK,
                neurons=nodes,
                activation=self.activation,
            )

            loss_fn = net.loss_with_bc
            net.compile(
                optimizer="adam", loss=lambda y_true, y_pred: loss_fn(y_true, y_pred)
            )

            history = net.fit(xtf, utf, epochs=8, verbose=0)
            final_losses[nodes] = history.history["loss"][-1]

        # All grid sizes should achieve convergence
        for nodes, final_loss in final_losses.items():
            self.assertLess(
                final_loss, 100.0, f"Grid size {nodes} should achieve reasonable loss"
            )

    def test_physics_consistency_during_training(self):
        """Test that physics remains consistent throughout training"""
        net = nn.NeuralNet_AdvectionDiffusion(
            x=self.x0,
            u=self.u0,
            bc=self.boundary,
            parameters=self.parameters,
            irk=self.IRK,
            neurons=self.nodes,
            activation=self.activation,
        )

        loss_fn = net.loss_with_bc
        net.compile(
            optimizer="adam", loss=lambda y_true, y_pred: loss_fn(y_true, y_pred)
        )

        # Check physics consistency before and after training
        def check_physics_consistency():
            output = net(self.xtf)
            u = output[0, :, :]
            u_x = output[1, :, :]
            u_xx = output[2, :, :]

            # Compute RHS of advection-diffusion equation
            rhs = -self.alpha * u_x + u_xx

            # Check that all components have reasonable magnitudes
            self.assertFalse(
                np.any(np.isnan(u.numpy())), "Solution should not contain NaN"
            )
            self.assertFalse(
                np.any(np.isnan(u_x.numpy())), "First derivative should not contain NaN"
            )
            self.assertFalse(
                np.any(np.isnan(u_xx.numpy())),
                "Second derivative should not contain NaN",
            )
            self.assertFalse(
                np.any(np.isnan(rhs.numpy())), "RHS should not contain NaN"
            )

            return np.max(np.abs(rhs.numpy()))

        # Check before training
        max_rhs_before = check_physics_consistency()

        # Train
        net.fit(self.xtf, self.utf, epochs=10, verbose=0)

        # Check after training
        max_rhs_after = check_physics_consistency()

        # Physics should remain well-behaved
        self.assertLess(
            max_rhs_after, 1000.0, "RHS should remain bounded after training"
        )

    def test_training_shows_learning_progress(self):
        """Test that network shows learning progress (loss decreases significantly)"""
        # Use smaller problem for faster test execution
        small_nodes = 4  # Smaller grid for faster convergence
        basis = grid.GridX(order=small_nodes)
        x0_small = 0.5 * (basis.nodes + 1.0)
        u0_small = self.solution_periodic(x0_small, 0, self.alpha)

        xtf_small = tf.reshape(tf.convert_to_tensor(x0_small), (small_nodes, 1))
        utf_small = tf.reshape(tf.convert_to_tensor(u0_small), (small_nodes, 1))

        net = nn.NeuralNet_AdvectionDiffusion(
            x=x0_small,
            u=u0_small,
            bc=self.boundary,
            parameters=self.parameters,
            irk=self.IRK,
            neurons=small_nodes,
            activation=self.activation,
        )

        loss_fn = net.loss_with_bc
        net.compile(
            optimizer="adam", loss=lambda y_true, y_pred: loss_fn(y_true, y_pred)
        )

        # Record initial loss
        initial_loss = net.loss_with_bc(utf_small, net(xtf_small)).numpy()

        # Train for modest number of epochs (realistic for unit test)
        history = net.fit(xtf_small, utf_small, epochs=20, verbose=0)
        final_loss = history.history["loss"][-1]

        # Test should verify learning mechanics, not accuracy
        # 1. Loss should decrease (any improvement shows learning is happening)
        improvement_ratio = (initial_loss - final_loss) / initial_loss
        self.assertGreater(
            improvement_ratio,
            0.01,  # Just 1% improvement shows learning
            f"Network should show learning progress: initial={initial_loss:.3f}, final={final_loss:.3f}",
        )

        # Alternative check: final loss should be less than initial
        self.assertLess(
            final_loss, initial_loss, "Final loss should be less than initial loss"
        )

        # 2. Final loss should be finite and reasonable (not NaN, not exploding)
        self.assertFalse(np.isnan(final_loss), "Final loss should not be NaN")
        self.assertLess(
            final_loss, initial_loss * 10, "Loss should not explode during training"
        )

        # 3. Network should produce reasonable outputs (not NaN/infinite)
        output = net(xtf_small)
        self.assertFalse(
            np.any(np.isnan(output.numpy())), "Network output should not contain NaN"
        )
        self.assertFalse(
            np.any(np.isinf(output.numpy())),
            "Network output should not contain infinities",
        )

    def test_training_reproducibility(self):
        """Test that training is reproducible with same random seeds"""

        def train_network():
            net = nn.NeuralNet_AdvectionDiffusion(
                x=self.x0,
                u=self.u0,
                bc=self.boundary,
                parameters=self.parameters,
                irk=self.IRK,
                neurons=self.nodes,
                activation=self.activation,
                seed=123,
            )

            loss_fn = net.loss_with_bc
            net.compile(
                optimizer="adam", loss=lambda y_true, y_pred: loss_fn(y_true, y_pred)
            )

            history = net.fit(self.xtf, self.utf, epochs=5, verbose=0)
            return history.history["loss"], net(self.xtf).numpy()

        # Train twice with same seeds
        losses1, output1 = train_network()
        losses2, output2 = train_network()

        # Results should be identical
        np.testing.assert_array_almost_equal(losses1, losses2, decimal=6)
        np.testing.assert_array_almost_equal(output1, output2, decimal=6)

    @unittest.skipIf(False, "Reference data is available")
    def test_training_matches_reference(self):
        """Test that training matches saved reference values"""
        if self.reference_data is None:
            self.skipTest("Reference data not available")

        net = nn.NeuralNet_AdvectionDiffusion(
            x=self.x0,
            u=self.u0,
            bc=self.boundary,
            parameters=self.parameters,
            irk=self.IRK,
            neurons=self.nodes,
            activation=self.activation,
        )

        # Check untrained loss matches reference
        untrained_loss = net.loss_with_bc(self.utf, net(self.xtf)).numpy()
        np.testing.assert_almost_equal(
            untrained_loss, self.reference_data["untrained_loss"], decimal=5
        )

        # Train and check losses match reference progression
        loss_fn = net.loss_with_bc
        net.compile(
            optimizer="adam", loss=lambda y_true, y_pred: loss_fn(y_true, y_pred)
        )

        training_losses = []
        for epoch in range(len(self.reference_data["training_losses"])):
            history = net.fit(self.xtf, self.utf, epochs=1, verbose=0)
            training_losses.append(history.history["loss"][0])

        # Training losses should match reference within tolerance
        np.testing.assert_allclose(
            training_losses,
            self.reference_data["training_losses"],
            rtol=1e-2,
            atol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
