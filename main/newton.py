import numpy as np
import numpy.linalg as linalg


# For lorenz system
def lorenz_dynamics(q):
    x, y, z = q[:, 0], q[:, 1], q[:, 2]

    sigma = 10.0
    beta = 8.0 / 3.0
    rho = 28.0

    f_rhs = np.array([[sigma * (y[i] - x[i]),
                      x[i] * (rho - z[i]) - y[i],
                      x[i] * y[i] - beta * z[i]] for i in range(q.shape[0])])

    jacobian = np.array([[[-sigma, sigma, 0],
                         [rho, 1.0, -x[i]],
                         [y[i], x[i], -beta]] for i in range(q.shape[0])])

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
        rhs_vec, _ = lorenz_dynamics(q + dt * np.matmul(irk.rk_matrix, rhs_in))  # (a11 * k1 + a12 * k2))
        error_vec = rhs_in - rhs_vec
        return error_vec

    def err_norm(err_in):
        return np.sqrt(np.square(err_in).sum())

    # Newton iteration
    itr = 0
    error = err_norm(err(k_vec))
    while error > threshold and itr < max_iterations:
        # Iterate of stages
        _, j_vec = lorenz_dynamics(q + dt * np.matmul(irk.rk_matrix, k_vec))  # (a11 * k1 + a12 * k2))
        # Jacobian of IRK method, I - dt * A * J with J jacobian of original system
        jac = (np.tensordot(np.eye(irk.order), np.eye(q.shape[0]), axes=0) -
               dt * np.einsum('ij,ikl->ijkl', irk.rk_matrix, j_vec))

        # Reshape to a single square system
        err_m = err(k_vec)
        jac = np.transpose(jac, axes=[0, 2, 1, 3])
        jac_r = jac.reshape(irk.order * q.shape[0], irk.order * q.shape[0])
        err_r = err_m.reshape(irk.order * q.shape[0])
        solution = linalg.solve(jac_r, err_r).reshape(irk.order, q.shape[0])

        # Get iterate k's
        damping = 0.25
        k_vec -= damping * solution

        # Error
        error = err_norm(err(k_vec))

        # print('\nNewton iteration ' + str(itr) + ' with err ' + str(error))
        itr += 1

        if itr >= max_iterations:
            print('Did not converge by iterations!')
            return k_vec

    print('Newton iteration took ' + str(itr) + ' tries, with error %.3e' % error)
    return k_vec
