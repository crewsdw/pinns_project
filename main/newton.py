import numpy as np
import numpy.linalg as linalg
from scipy.optimize import root


# For lorenz system
def lorenz_dynamics(q):
    x, y, z = q[:, 0], q[:, 1], q[:, 2]

    sigma = 10.0
    beta = 8.0 / 3.0
    rho = 28.0

    f_rhs = np.array(
        [
            [
                sigma * (y[i] - x[i]),
                x[i] * (rho - z[i]) - y[i],
                x[i] * y[i] - beta * z[i],
            ]
            for i in range(q.shape[0])
        ]
    )

    jacobian = np.array(
        [
            [[-sigma, sigma, 0], [rho, 1.0, -x[i]], [y[i], x[i], -beta]]
            for i in range(q.shape[0])
        ]
    )

    return f_rhs, jacobian


# IRK update
def newton_irk(q, dt, irk, threshold, max_iterations, guess):
    # position guess: explicit step
    # rhs, _ = lorenz_dynamics(q[None, :])
    # rhs = rhs[0, :]
    # q1 = q + 0.5 * (1.0 - 1.0 / np.sqrt(3.0)) * dt * rhs
    # q2 = q + 0.5 * (1.0 + 1.0 / np.sqrt(3.0)) * dt * rhs
    # position guess: neural net output
    guess = guess[:, :-1].transpose()

    # Iterate of stages
    k_vec, j_vec = lorenz_dynamics(guess)

    def err(rhs_in):
        rhs_vec, _ = lorenz_dynamics(
            q + dt * np.matmul(irk.rk_matrix, rhs_in)
        )  # (a11 * k1 + a12 * k2))
        error_vec = rhs_in - rhs_vec
        return error_vec

    def err_norm(err_in):
        return np.sqrt(np.square(err_in).sum())

    # Newton iteration
    itr = 0
    error = err_norm(err(k_vec))
    while error > threshold and itr < max_iterations:
        # Iterate of stages
        _, j_vec = lorenz_dynamics(
            q + dt * np.matmul(irk.rk_matrix, k_vec)
        )  # (a11 * k1 + a12 * k2))
        # Jacobian of IRK method, I - dt * A * J with J jacobian of original system
        jac = np.tensordot(
            np.eye(irk.order), np.eye(q.shape[0]), axes=0
        ) - dt * np.einsum("ij,ikl->ijkl", irk.rk_matrix, j_vec)

        # Reshape to a single square system
        err_m = err(k_vec)
        jac = np.transpose(jac, axes=[0, 2, 1, 3])
        jac_r = jac.reshape(irk.order * q.shape[0], irk.order * q.shape[0])
        err_r = err_m.reshape(irk.order * q.shape[0])
        solution = linalg.solve(jac_r, err_r).reshape(irk.order, q.shape[0])

        # Get iterate k's
        damping = 0.8
        k_vec -= damping * solution

        # Error
        error = err_norm(err(k_vec))

        # print('\nNewton iteration ' + str(itr) + ' with err ' + str(error))
        itr += 1

        if itr >= max_iterations:
            print("Did not converge by max iterations!")
            print(f"Error is {error:.3e}")
            return k_vec

    print("Newton iteration took " + str(itr) + " tries, with error %.3e" % error)
    return k_vec


# Adaptive IRK update using scipy.optimize.root
def newton_irk_adaptive(q, dt, irk, threshold, max_iterations, guess, method='hybr'):
    """
    Adaptive Newton method using scipy.optimize.root with automatic line search.

    Parameters:
    - method: 'hybr' (Powell hybrid), 'lm' (Levenberg-Marquardt), 'broyden1', etc.
    """
    # Initial guess from neural net output
    guess = guess[:, :-1].transpose()

    def residual_function(k_vec_flat):
        """Residual function for the IRK system."""
        k_vec = k_vec_flat.reshape(irk.order, q.shape[0])

        # Compute RHS at updated positions
        rhs_vec, _ = lorenz_dynamics(
            q + dt * np.matmul(irk.rk_matrix, k_vec)
        )

        # IRK residual: k - f(q + dt * A * k) = 0
        error_vec = k_vec - rhs_vec
        return error_vec.flatten()

    def jacobian_function(k_vec_flat):
        """Jacobian of the residual function."""
        k_vec = k_vec_flat.reshape(irk.order, q.shape[0])

        # Get Jacobian of original system
        _, j_vec = lorenz_dynamics(
            q + dt * np.matmul(irk.rk_matrix, k_vec)
        )

        # Jacobian of IRK residual: I - dt * A * J
        jac = np.tensordot(
            np.eye(irk.order), np.eye(q.shape[0]), axes=0
        ) - dt * np.einsum("ij,ikl->ijkl", irk.rk_matrix, j_vec)

        # Reshape to matrix form
        jac = np.transpose(jac, axes=[0, 2, 1, 3])
        return jac.reshape(irk.order * q.shape[0], irk.order * q.shape[0])

    # Flatten initial guess
    k0 = guess.flatten()

    # Method-specific options for better convergence
    if method == 'lm':
        options = {
            'ftol': threshold,  # Function tolerance
            'xtol': threshold,  # Solution tolerance
            'gtol': threshold,  # Gradient tolerance
            'maxiter': max_iterations,
        }
    elif method == 'hybr':
        options = {
            'xtol': threshold,
            'maxiter': max_iterations,
        }
    else:
        options = {
            'xtol': threshold,
            'maxiter': max_iterations,
        }

    if method in ['hybr', 'lm']:
        # These methods can use the Jacobian
        result = root(residual_function, k0, method=method, jac=jacobian_function, options=options)
    else:
        # Jacobian-free methods
        result = root(residual_function, k0, method=method, options=options)

    final_residual_norm = np.linalg.norm(result.fun)

    # Check if we actually achieved the desired tolerance
    actually_converged = final_residual_norm < threshold * 10  # Give some leeway

    if result.success and actually_converged:
        print(f"Adaptive Newton ({method}) converged in {result.nfev} function evaluations")
        print(f"Final residual norm: {final_residual_norm:.3e}")
    elif result.success and not actually_converged:
        print(f"Adaptive Newton ({method}) claimed convergence but residual too large!")
        print(f"Final residual norm: {final_residual_norm:.3e} (threshold: {threshold:.3e})")
        print(f"Message: {result.message}")
    else:
        print(f"Adaptive Newton ({method}) failed to converge: {result.message}")
        print(f"Final residual norm: {final_residual_norm:.3e}")

    # Return k_vec in original shape
    return result.x.reshape(irk.order, q.shape[0])
