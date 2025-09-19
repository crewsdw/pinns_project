# Physics-Informed Neural Networks (PINNs) for Implicit Runge-Kutta Methods

This project implements Physics-Informed Neural Networks (PINNs) to learn implicit Runge-Kutta (IRK) time-stepping schemes for differential equations, specifically targeting the Lorenz system and advection-diffusion equations.

## Overview

Traditional numerical methods for solving differential equations require explicit computation of intermediate stages in implicit Runge-Kutta schemes, often involving expensive iterative solvers like Newton's method. This project explores using neural networks to directly predict these intermediate stages, potentially accelerating time integration.

### Key Innovation

Instead of solving the nonlinear system:
```
k_i = f(y_n + dt * Σ a_ij * k_j)  for i = 1,...,s
```

The neural network learns to predict all stages `{k_1, k_2, ..., k_s}` simultaneously, which are then refined using Newton iteration for high accuracy.

## Mathematical Framework

### Implicit Runge-Kutta Method

For an ODE `dy/dt = f(y)`, the IRK method advances the solution via:
```
y_{n+1} = y_n + dt * Σ b_i * k_i
k_i = f(y_n + dt * Σ a_ij * k_j)
```

Where `A = [a_ij]` is the Runge-Kutta coefficient matrix and `b` are the quadrature weights.

### Physics-Informed Loss Functions

The networks are trained using physics-based loss functions that enforce the underlying differential equations:

**Lorenz System**: Neural network predicts RK stages `u1` for the Lorenz equations:
```
dx/dt = σ(y - x)
dy/dt = x(ρ - z) - y
dz/dt = xy - βz
```

Loss function: `||u0_true - (u1 - dt * RHS × A)||²`

**Advection-Diffusion**: Neural network predicts solution and spatial derivatives for:
```
∂u/∂t + α∂u/∂x = ∂²u/∂x²
```

Loss function includes PDE residual and boundary condition enforcement.

## Project Structure

```
main/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── main.py                     # Main execution script
├── neural_net.py              # Neural network definitions
├── irk_coefficients.py        # IRK coefficient computation
├── x_grid.py                  # Spatial grid generation
├── newton.py                  # Newton solver for refinement
└── tests/                     # Comprehensive test suite
    ├── README.md
    ├── run_tests.py
    ├── generate_reference_data.py
    ├── test_network_initialization.py
    ├── test_lorenz_regression.py
    ├── test_advection_diffusion_regression.py
    ├── test_lorenz_training.py
    └── test_advection_diffusion_training.py
```

## Neural Network Architectures

### NeuralNet_LorenzStepper
- **Input**: Current state `[x, y, z]`
- **Output**: All IRK stages `(3, order+1)` tensor
- **Loss**: Physics-informed loss enforcing Lorenz dynamics
- **Training**: Each time step trains a fresh network

### NeuralNet_AdvectionDiffusion
- **Input**: Spatial coordinates `x`
- **Output**: Solution `u`, first derivative `∂u/∂x`, second derivative `∂²u/∂x²`
- **Loss**: PDE residual + boundary conditions
- **Training**: Single network for spatial problem

## Key Features

### High-Order Accuracy
- Supports IRK methods up to order 100+ using Gauss-Legendre nodes
- Maintains spectral accuracy for smooth problems

### Physics Enforcement
- Loss functions directly encode differential equation physics
- Automatic differentiation computes spatial derivatives
- Boundary conditions enforced through penalty methods

### Hybrid Approach
- Neural network provides initial guess for Newton iteration
- Combines ML acceleration with guaranteed convergence
- Maintains numerical accuracy of traditional methods

## Usage

### Basic Execution
```bash
# Activate conda environment
conda activate pinn

# Run main simulation
python main.py
```

### Running Tests
```bash
# Run all tests
cd tests/
python run_tests.py

# Generate reference data first
python generate_reference_data.py

# Run specific test categories
python run_tests.py test_lorenz_regression.py
```

## Parameters and Configuration

### Lorenz System Parameters
```python
order = 100        # IRK method order
dt = 0.8          # Time step size
epochs = 12000    # Training epochs per time step
activation = 'elu' # Neural network activation
optimizer = 'adam' # Training optimizer
```

### Advection-Diffusion Parameters
```python
nodes = 32        # Spatial grid points
dt = 0.05         # Time step size
alpha = 1.0       # Advection coefficient
epochs = 1500     # Training epochs
activation = 'tanh'
```

## Research Applications

### Computational Efficiency
- Reduces computational cost of implicit time integration
- Particularly beneficial for stiff differential equations
- Potential for real-time applications requiring fast time stepping

### Scientific Computing
- Weather and climate modeling (advection-diffusion processes)
- Fluid dynamics (implicit time integration)
- Molecular dynamics (stiff chemical kinetics)
- Astrophysics (N-body problems with implicit methods)

### Machine Learning
- Novel application of PINNs to numerical analysis
- Demonstrates physics-constrained learning
- Hybrid numerical-ML methodology

## Technical Implementation

### IRK Coefficient Generation
- Gauss-Legendre quadrature nodes and weights
- Vandermonde matrix construction for high-order accuracy
- Efficient TensorFlow implementation for GPU acceleration

### Automatic Differentiation
- TensorFlow's `tf.gradients` computes spatial derivatives
- Enables exact enforcement of PDE physics
- Eliminates finite difference approximation errors

### Training Strategy
- Fresh network per time step for Lorenz system
- Single network training for spatial problems
- Early stopping and learning rate scheduling

## Future Directions

### Framework Migration
- Current implementation uses TensorFlow
- Comprehensive test suite enables PyTorch migration
- Regression tests ensure numerical consistency

### Extensions
- Multi-dimensional PDEs (2D/3D advection-diffusion)
- Nonlinear PDE systems
- Adaptive time stepping with neural networks
- Transfer learning between similar problems

### Performance Optimization
- GPU acceleration and distributed training
- Model compression for deployment
- Real-time inference optimization

## Dependencies

- **NumPy**: Numerical computations and array operations
- **TensorFlow**: Neural network implementation and automatic differentiation
- **SciPy**: Special functions for IRK coefficient computation
- **Matplotlib**: Visualization and result plotting

## Citation

If you use this code in your research, please cite:

```bibtex
@software{pinn_irk_2024,
  title={Physics-Informed Neural Networks for Implicit Runge-Kutta Methods},
  author={Your Name},
  year={2024},
  url={https://github.com/yourname/pinns_project}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## Acknowledgments

- Implicit Runge-Kutta theory and implementation
- Physics-Informed Neural Networks methodology
- TensorFlow team for automatic differentiation capabilities