import numpy as np
import numpy.linalg as linalg


# For lorenz system
def lorenz_dynamics(q):
    x, y, z = q[:, 0], q[:, 1], q[:, 2]
    # print(q.shape)
    # quit()
    sigma = 10
    beta = 8 / 3
    rho = 28

    # f_rhs = np.zeros((q.shape[0], 3, 3))
    #
    # if x.shape
    # f_rhs = np.array([sigma * (y - x),
    #                   x * (rho - z) - y,
    #                   x * y - beta * z])
    f_rhs = np.array([[sigma * (y[i] - x[i]),
                      x[i] * (rho - z[i]) - y[i],
                      x[i] * y[i] - beta * z[i]] for i in range(q.shape[0])])
    # print(f_rhs.shape)
    # quit()

    jacobian = np.array([[[-sigma, sigma, 0],
                         [rho, 1.0, -x[i]],
                         [y[i], x[i], -beta]] for i in range(q.shape[0])])

    return f_rhs, jacobian


# IRK update
def newton_irk(q, guess, dt, irk, threshold, max_iterations):
    # Use an explicit Euler step as initial guess
    # rhs, _ = lorenz_dynamics(q[None, :])
    # rhs = rhs[0, :]

    # position guess: explicit step
    # q1 = q + 0.5 * (1.0 - 1.0 / np.sqrt(3.0)) * dt * rhs
    # q2 = q + 0.5 * (1.0 + 1.0 / np.sqrt(3.0)) * dt * rhs
    # print(guess.shape)
    # guess = q + dt * np.tensordot((irk.nodes + 1.0)/2.0, rhs, axes=0)
    # print(guess.shape)
    # quit()
    guess = guess[:, :-1].transpose()
    # print(q_guess.shape)
    # IRK matrix
    # a11, a12 = 0.25, 0.25 - np.sqrt(3.0) / 6.0
    # a21, a22 = 0.25 + np.sqrt(3.0) / 6.0, 0.25
    # a_mtx = irk.rk_matrix

    # Iterate of stages
    k_vec, j_vec = lorenz_dynamics(guess)
    # print(k_vec.shape)
    # quit()

    def err(rhs_in):
        # print(k_in.shape)
        # print(irk.rk_matrix.shape)
        # quit()
        rhs_vec, _ = lorenz_dynamics(q + dt * np.matmul(irk.rk_matrix, rhs_in))  # (a11 * k1 + a12 * k2))
        error_vec = rhs_in - rhs_vec
        # error2 = k2 - rhs2
        # return np.array([error1, error2])
        return error_vec

    def err_norm(err_in):
        return np.sqrt(np.square(err_in).sum())

    itr = 0
    error = err_norm(err(k_vec))
    while error > threshold and itr < max_iterations:
        # Iterate of stages
        _, j_vec = lorenz_dynamics(q + dt * np.matmul(irk.rk_matrix, k_vec))  # (a11 * k1 + a12 * k2))
        # print(j_vec.shape)
        # Jacobian of IRK method, I - dt * A * J with J jacobian of original system
        # jac = np.array([[np.eye(q.shape[0]) - dt * a11 * j1, -dt * a12 * j1],
        #                 [-dt * a21 * j2, np.eye(q.shape[0]) - dt * a22 * j2]])
        # jac = (np.tensordot(np.eye(irk.order), np.eye(q.shape[0]), axes=0) -
        #        dt * np.tensordot(irk.rk_matrix, j_vec, axes=0))
        jac = (np.tensordot(np.eye(irk.order), np.eye(q.shape[0]), axes=0) -
               dt * np.einsum('ij,ikl->ijkl', irk.rk_matrix, j_vec))

        # Newton iteration
        # Reshape to a single square system
        err_m = err(k_vec)
        jac = np.transpose(jac, axes=[0, 2, 1, 3])
        jac_r = jac.reshape(irk.order * q.shape[0], irk.order * q.shape[0])
        err_r = err_m.reshape(irk.order * q.shape[0])
        # k_next = np.array([k1, k2]) - linalg.solve(jac, err(k1, k2)[None, :, :])
        # solve = linalg.solve(jac, err(k1, k2)[None, :, None, :])
        solution = linalg.solve(jac_r, err_r).reshape(irk.order, q.shape[0])
        # print(solution.shape)
        # print(k_vec.shape)
        # quit()
        # Get iterate k's
        damping = 0.25
        k_vec -= damping * solution
        # k1 = k1 - solution[0, :]
        # k2 = k2 - solution[1, :]

        # Error
        error = err_norm(err(k_vec))

        # print('\nNewton iteration ' + str(itr) + ' with err ' + str(error))
        itr += 1

        if itr >= max_iterations:
            print('Did not converge by iterations!')
            return k_vec

    print('Newton iteration took ' + str(itr) + ' tries, with error %.3e' % error)

    return k_vec
