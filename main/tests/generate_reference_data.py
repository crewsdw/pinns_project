#!/usr/bin/env python3
"""
Generate reference data for regression tests.
Run this script to create saved reference values for network outputs and training.
"""

import os
import pickle
import sys

import numpy as np
import tensorflow as tf

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import irk_coefficients as irk
import neural_net as nn
import x_grid as grid


def generate_lorenz_reference_data():
    """Generate reference data for Lorenz system"""
    print("Generating Lorenz reference data...")

    # Set reproducible seeds
    np.random.seed(42)
    tf.random.set_seed(42)

    # Parameters
    order = 4
    IRK = irk.IRK(order=order)
    IRK.build_matrix()

    dt = 0.1
    alpha = 1.0
    parameters = [dt, alpha]
    activation = "tanh"

    # Initial condition
    q = np.array([1.0, 1.0, 1.0])
    u0 = np.array([q])
    utf = tf.reshape(tf.convert_to_tensor(u0), (u0.shape[0], u0.shape[1]))

    # Create network
    net = nn.NeuralNet_LorenzStepper(
        parameters=parameters, irk=IRK, neurons=q.shape[0], activation=activation
    )

    # Get untrained output
    untrained_output = net(utf).numpy()

    # Get untrained loss
    untrained_loss = net.custom_loss(utf, net(utf)).numpy()

    # Train for a few epochs to get training progression
    loss_fn = net.custom_loss
    net.compile(optimizer="adam", loss=lambda y_true, y_pred: loss_fn(y_true, y_pred))

    # Collect training history
    training_losses = []
    for epoch in range(5):  # Just a few epochs for reference
        history = net.fit(utf, utf, epochs=1, verbose=0)
        training_losses.append(history.history["loss"][0])

    trained_output = net(utf).numpy()

    reference_data = {
        "parameters": {
            "order": order,
            "dt": dt,
            "alpha": alpha,
            "activation": activation,
            "initial_condition": q,
        },
        "untrained_output": untrained_output,
        "untrained_loss": untrained_loss,
        "training_losses": training_losses,
        "trained_output": trained_output,
        "irk_matrix": IRK.rk_matrix_tf32.numpy(),
        "irk_weights": IRK.weights,
    }

    return reference_data


def generate_advection_diffusion_reference_data():
    """Generate reference data for Advection-Diffusion system"""
    print("Generating Advection-Diffusion reference data...")

    # Set reproducible seeds
    np.random.seed(42)
    tf.random.set_seed(42)

    # Parameters
    order = 4
    nodes = 8
    IRK = irk.IRK(order=order)
    IRK.build_matrix()

    dt = 0.05
    alpha = 1.0
    parameters = [dt, alpha]
    activation = "tanh"

    # Create grid and initial condition
    basis = grid.GridX(order=nodes)
    x0 = 0.5 * (basis.nodes + 1.0)

    def solution_periodic(x, t, a):
        return np.sin(2.0 * np.pi * (x - a * t)) * np.exp(-((2.0 * np.pi) ** 2.0) * t)

    u0 = solution_periodic(x0, 0, alpha)
    boundary = np.array([0.0, 1.0])

    # Convert to tensors
    xtf = tf.reshape(tf.convert_to_tensor(x0), (nodes, 1))
    utf = tf.reshape(tf.convert_to_tensor(u0), (nodes, 1))

    # Create network
    net = nn.NeuralNet_AdvectionDiffusion(
        x=x0,
        u=u0,
        bc=boundary,
        parameters=parameters,
        irk=IRK,
        neurons=nodes,
        activation=activation,
    )

    # Get untrained output
    untrained_output = net(xtf).numpy()

    # Get untrained loss
    untrained_loss = net.loss_with_bc(utf, net(xtf)).numpy()

    # Train for a few epochs
    loss_fn = net.loss_with_bc
    net.compile(optimizer="adam", loss=lambda y_true, y_pred: loss_fn(y_true, y_pred))

    training_losses = []
    for epoch in range(5):
        history = net.fit(xtf, utf, epochs=1, verbose=0)
        training_losses.append(history.history["loss"][0])

    trained_output = net(xtf).numpy()

    # Analytical solution at next time step for comparison
    u1_analytical = solution_periodic(x0, dt, alpha)

    reference_data = {
        "parameters": {
            "order": order,
            "nodes": nodes,
            "dt": dt,
            "alpha": alpha,
            "activation": activation,
        },
        "grid": {"x0": x0, "u0": u0, "boundary": boundary},
        "untrained_output": untrained_output,
        "untrained_loss": untrained_loss,
        "training_losses": training_losses,
        "trained_output": trained_output,
        "analytical_solution_next_step": u1_analytical,
        "irk_matrix": IRK.rk_matrix_tf32.numpy(),
        "irk_weights": IRK.weights,
    }

    return reference_data


def main():
    """Generate all reference data"""
    print("Generating reference data for regression tests...")

    # Create reference data directory
    ref_dir = os.path.join(os.path.dirname(__file__), "reference_data")
    os.makedirs(ref_dir, exist_ok=True)

    # Generate and save Lorenz reference data
    lorenz_data = generate_lorenz_reference_data()
    with open(os.path.join(ref_dir, "lorenz_reference.pkl"), "wb") as f:
        pickle.dump(lorenz_data, f)

    # Generate and save Advection-Diffusion reference data
    ad_data = generate_advection_diffusion_reference_data()
    with open(os.path.join(ref_dir, "advection_diffusion_reference.pkl"), "wb") as f:
        pickle.dump(ad_data, f)

    print("Reference data generated successfully!")
    print(f"Lorenz untrained loss: {lorenz_data['untrained_loss']:.6f}")
    print(f"Lorenz final training loss: {lorenz_data['training_losses'][-1]:.6f}")
    print(f"AD untrained loss: {ad_data['untrained_loss']:.6f}")
    print(f"AD final training loss: {ad_data['training_losses'][-1]:.6f}")


if __name__ == "__main__":
    main()
