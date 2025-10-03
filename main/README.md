# Physics-Informed Neural Networks (PINNs) for Implicit Runge-Kutta Methods

This project implements Physics-Informed Neural Networks (PINNs) to learn implicit Runge-Kutta (IRK) time-stepping schemes for differential equations, specifically targeting the Lorenz system and advection-diffusion equations.

## Overview

Traditional numerical methods for solving differential equations require explicit computation of intermediate stages in implicit Runge-Kutta schemes, often involving expensive iterative solvers like Newton's method. This project explores using neural networks to directly predict these intermediate stages, potentially accelerating time integration.

### Algorithm Overview

To advance a system state **S** from time **t** to **t + Δt**:

1. **Generate Legendre Points**: Create ~100 Legendre-Gauss-Lobatto quadrature points within the interval (t, t+Δt)

2. **Neural Network Prediction**: Train a neural network to map the initial state **s ∈ S** to all intermediate solutions **s_i ∈ S×N** where **N** is the number of Legendre points. The network essentially approximates the integral of the differential equation over the time interval.

3. **Derivative Computation**: Compute derivatives at each solution point using Newton's method for high accuracy

4. **IRK Integration**: Insert the computed derivatives into the Implicit Runge-Kutta coefficient matrix to obtain the final solution at **t + Δt**

### Key Innovation

The neural network is retrained for each time step, learning to approximate the integral of the differential equation within each specific time interval. Rather than solving the traditional nonlinear system:
```
k_i = f(y_n + dt * Σ a_ij * k_j)  for i = 1,...,s
```

The network directly predicts all IRK stages `{k_1, k_2, ..., k_s}` simultaneously, which are then refined using Newton iteration for guaranteed accuracy.

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run main simulation
python main.py

# Run tests
pytest .
```