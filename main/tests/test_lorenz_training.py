import os
import pickle
import sys
import unittest

import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import irk_coefficients as irk
import neural_net as nn


class TestLorenzTraining(unittest.TestCase):
    """Regression tests for Lorenz system training procedure"""

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

        self.reference_data = None
        ref_path = os.path.join(
            os.path.dirname(__file__), "reference_data", "lorenz_reference.pkl"
        )
        if os.path.exists(ref_path):
            with open(ref_path, "rb") as f:
                self.reference_data = pickle.load(f)

    def test_training_loss_decreases(self):
        """Test that training loss decreases over epochs"""
        net = nn.NeuralNet_LorenzStepper(
            parameters=self.parameters,
            irk=self.IRK,
            neurons=self.q.shape[0],
            activation=self.activation,
        )

        loss_fn = net.custom_loss
        net.compile(
            optimizer="adam", loss=lambda y_true, y_pred: loss_fn(y_true, y_pred)
        )

        initial_loss = net.custom_loss(self.utf, net(self.utf)).numpy()

        history = net.fit(self.utf, self.utf, epochs=3, verbose=0)

        final_loss = history.history["loss"][-1]
        self.assertLess(final_loss, initial_loss, "Training loss should decrease")

        losses = history.history["loss"]
        self.assertLess(
            np.mean(losses[-2:]),
            np.mean(losses[:2]),
            "Average final losses should be less than average initial losses",
        )

    def test_training_convergence_stability(self):
        """Test that training converges to a stable solution"""
        net = nn.NeuralNet_LorenzStepper(
            parameters=self.parameters,
            irk=self.IRK,
            neurons=self.q.shape[0],
            activation=self.activation,
        )

        loss_fn = net.custom_loss
        net.compile(
            optimizer="adam", loss=lambda y_true, y_pred: loss_fn(y_true, y_pred)
        )

        history = net.fit(self.utf, self.utf, epochs=8, verbose=0)

        losses = history.history["loss"]

        final_losses = losses[-5:]
        loss_variance = np.var(final_losses)
        self.assertLess(loss_variance, 0.01, "Loss should stabilize during training")

    def test_different_optimizers_converge(self):
        """Test that different optimizers can train the network"""
        optimizers = ["adam", "sgd"]  # Reduced for performance
        final_losses = {}

        for opt_name in optimizers:
            np.random.seed(42)
            tf.random.set_seed(42)

            net = nn.NeuralNet_LorenzStepper(
                parameters=self.parameters,
                irk=self.IRK,
                neurons=self.q.shape[0],
                activation=self.activation,
            )

            loss_fn = net.custom_loss
            net.compile(
                optimizer=opt_name, loss=lambda y_true, y_pred: loss_fn(y_true, y_pred)
            )

            # Train
            history = net.fit(self.utf, self.utf, epochs=3, verbose=0)
            final_losses[opt_name] = history.history["loss"][-1]

        for opt_name, final_loss in final_losses.items():
            self.assertLess(
                final_loss, 10.0, f"Optimizer {opt_name} should achieve reasonable loss"
            )

    def test_training_reproducibility(self):
        """Test that training is reproducible with same random seeds"""

        def train_network():
            net = nn.NeuralNet_LorenzStepper(
                parameters=self.parameters,
                irk=self.IRK,
                neurons=self.q.shape[0],
                activation=self.activation,
                seed=123,
            )

            loss_fn = net.custom_loss
            net.compile(
                optimizer="adam", loss=lambda y_true, y_pred: loss_fn(y_true, y_pred)
            )

            history = net.fit(self.utf, self.utf, epochs=3, verbose=0)
            return history.history["loss"], net(self.utf).numpy()

        losses1, output1 = train_network()
        losses2, output2 = train_network()

        np.testing.assert_array_almost_equal(losses1, losses2, decimal=6)
        np.testing.assert_array_almost_equal(output1, output2, decimal=6)

    def test_learning_rate_sensitivity(self):
        """Test training behavior with different learning rates"""
        learning_rates = [0.001, 0.01]  # Reduced for performance
        final_losses = {}

        for lr in learning_rates:
            np.random.seed(42)
            tf.random.set_seed(42)

            net = nn.NeuralNet_LorenzStepper(
                parameters=self.parameters,
                irk=self.IRK,
                neurons=self.q.shape[0],
                activation=self.activation,
            )

            loss_fn = net.custom_loss
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            net.compile(
                optimizer=optimizer, loss=lambda y_true, y_pred: loss_fn(y_true, y_pred)
            )

            history = net.fit(self.utf, self.utf, epochs=3, verbose=0)
            final_losses[lr] = history.history["loss"][-1]

        for lr, final_loss in final_losses.items():
            self.assertLess(
                final_loss, 100.0, f"Learning rate {lr} should achieve convergence"
            )

    def test_network_weights_update_during_training(self):
        """Test that network weights actually change during training"""
        net = nn.NeuralNet_LorenzStepper(
            parameters=self.parameters,
            irk=self.IRK,
            neurons=self.q.shape[0],
            activation=self.activation,
        )

        initial_weights = [w.copy() for w in net.get_weights()]

        loss_fn = net.custom_loss
        net.compile(
            optimizer="adam", loss=lambda y_true, y_pred: loss_fn(y_true, y_pred)
        )

        # Train
        net.fit(self.utf, self.utf, epochs=3, verbose=0)

        final_weights = net.get_weights()

        for initial, final in zip(initial_weights, final_weights):
            self.assertFalse(
                np.allclose(initial, final, atol=1e-6),
                "Network weights should change during training",
            )

    @unittest.skipIf(False, "Reference data is available")
    def test_training_matches_reference(self):
        """Test that training matches saved reference values"""
        if self.reference_data is None:
            self.skipTest("Reference data not available")

        net = nn.NeuralNet_LorenzStepper(
            parameters=self.parameters,
            irk=self.IRK,
            neurons=self.q.shape[0],
            activation=self.activation,
        )

        untrained_loss = net.custom_loss(self.utf, net(self.utf)).numpy()
        np.testing.assert_almost_equal(
            untrained_loss, self.reference_data["untrained_loss"], decimal=6
        )

        loss_fn = net.custom_loss
        net.compile(
            optimizer="adam", loss=lambda y_true, y_pred: loss_fn(y_true, y_pred)
        )

        training_losses = []
        for epoch in range(len(self.reference_data["training_losses"])):
            history = net.fit(self.utf, self.utf, epochs=1, verbose=0)
            training_losses.append(history.history["loss"][0])

        np.testing.assert_allclose(
            training_losses,
            self.reference_data["training_losses"],
            rtol=1e-3,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
