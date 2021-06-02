import numpy as np
import irk_coefficients as irk
import neural_net as nn
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import x_grid as grid
import newton as newton

keras.backend.clear_session()


# Problem to solve... first mode of linear advection-diffusion
def solution_dirichlet(x, t, a):
    return np.exp(a * 0.5 * (x - a * 0.5 * t)) * np.sin(np.pi * x) * np.exp(-(np.pi ** 2.0) * t)


def solution_periodic(x, t, a):
    return np.sin(2.0 * np.pi * (x - a * t)) * np.exp(-(2.0 * np.pi) ** 2.0 * t)


# Initialize IRK coefficient matrix...
order = 100  # 50  # 10
nodes = 32
IRK = irk.IRK(order=order)
IRK.build_matrix()

dt = 0.8
# Net parameters
alpha = 1.0
parameters = [dt, alpha]
# neurons = nodes
activation = 'elu'  # 'tanh'
optimizer = 'adam'
epochs = 12000

# Main net loop for Lorenz system
steps = 5
q = np.array([10.54, -4.112, 35.82])
# q = np.array([1.0, -1.0, -1.0])
q_loop = np.zeros((3, order+1, steps))
for idx in range(steps):
    # Lorenz with IRK:

    # q1 = np.array([8, 3, 20])
    # q2 = np.array([6, 2, 10])
    # q = np.array([10.5, 4.0, 30.0])
    # dt = 0.1

    u0 = np.array([q])  # , q1, q2])

    # Make neural net
    # xtf = tf.reshape(tf.convert_to_tensor(x0), (nodes, 1))
    utf = tf.reshape(tf.convert_to_tensor(u0), (u0.shape[0], u0.shape[1]))
    net = nn.NeuralNet_LorenzStepper(parameters=parameters, irk=IRK, neurons=q.shape[0], activation=activation)
    net.compile(optimizer=optimizer, loss=net.loss)

    # Fit model
    net.fit(utf, utf, epochs=epochs, shuffle=True)  # , callbacks=[early_stop])  # , batch_size=nodes)

    out = net.predict(utf)  # , batch_size=nodes)

    sigma = 10
    beta = 8 / 3
    rho = 28
    rhs = tf.convert_to_tensor([sigma * (out[:, 1, :] - out[:, 0, :]),
                            out[:, 0, :] * (rho - out[:, 2, :]) - out[:, 1, :],
                            out[:, 0, :] * out[:, 1, :] - beta * out[:, 2, :]])
    u0_out = np.asarray(out - dt * tf.transpose(tf.matmul(rhs, IRK.rk_matrix_tf32), perm=(1, 0, 2)))
    # print(out.shape)

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(u0_out[:, 0, :].flatten(), u0_out[:, 1, :].flatten(), u0_out[:, 2, :].flatten(), label='predicted u0')
    # ax.scatter(u0[:, 0], u0[:, 1], u0[:, 2])
    # ax.scatter(out[:, 0, :].flatten(), out[:, 1, :].flatten(), out[:, 2, :].flatten(), label='predicted rk stages')
    # plt.show()

    # Use output as guess for Newton solver

    nt = 2

    qs = np.zeros((3, order+1, nt))
    for i in range(order+1):
        qs[:, i, 0] = q

    for i in range(1, nt):
        # Newton iteration for rhs evaluations
        k_vec = newton.newton_irk(q, guess=out[0, :, :], dt=dt, irk=IRK, threshold=1.0e-10, max_iterations=300)
        # GL stages
        print(k_vec.shape)
        print(IRK.rk_matrix.shape)
        qs[:, :-1, i] = q[:, None] + dt * np.transpose(np.tensordot(IRK.rk_matrix, k_vec, axes=([1], [0])), axes=([1, 0]))
        # Update
        q += 0.5 * dt * np.tensordot(IRK.weights, k_vec, axes=([0], [0]))  # 0.5 * dt * (k1 + k2)
        qs[:, -1, i] = q

    q_loop[:, :, idx] = qs[:, :, -1]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_title('Lorenz system implicit RK advance, dt=%.3e' % dt + ', err threshold 1.0e-10')
for i in range(steps):
    ax.scatter(q_loop[0, :, i], q_loop[1, :, i], q_loop[0, :, i], label='stages of time-step ' + str(i))
# ax.scatter(out[:, 0, :].flatten(), out[:, 1, :].flatten(), out[:, 2, :].flatten(), label='predicted rk stages')
# ax.scatter(qs[0, :, :].flatten(), qs[1, :, :].flatten(), qs[2, :, :].flatten(), label='newton iterated rk stages')
# plt.legend(loc='best')
# for i in range(nt):
#     ax.plot(qs[0, :, i], qs[1, :, i], qs[2, :, i], 'o--')
# print(qs)
plt.show()
print('show plz')
quit()

# Advection-Diffusion with net:
# Net parameters
dt = 0.05
alpha = 1.0
parameters = [dt, alpha]
# neurons = nodes
activation = 'tanh'
optimizer = 'adam'
epochs = 1500
# early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20)

# Make grid
basis = grid.GridX(order=nodes)
x0 = 0.5 * (basis.nodes + 1.0)
# x0 = np.linspace(0, 1, num=nodes)
# x0 = 0.5 * (np.array(IRK.nodes) + 1.0)  # GL nodes on [0,1]
# u0 = solution_dirichlet(x0, 0, alpha)
u0 = solution_periodic(x0, 0, alpha)

# Look at it...
plt.figure()
plt.plot(x0, u0, 'o--', label='Initial condition')
plt.legend(loc='best')
plt.grid(True)
plt.show()

# Boundary points
lb = 0.0
rb = 1.0
boundary = np.array([lb, rb])

# Make neural net
xtf = tf.reshape(tf.convert_to_tensor(x0), (nodes, 1))
utf = tf.reshape(tf.convert_to_tensor(u0), (nodes, 1))
net = nn.NeuralNet_AdvectionDiffusion(x=x0, u=u0, bc=boundary, parameters=parameters,
                                      irk=IRK, neurons=nodes, activation=activation)
net.compile(optimizer=optimizer, loss=net.loss_with_bc)  # loss=tf.keras.losses.MeanSquaredError())
# net.loss_with_bc)  # loss=tf.keras.losses.MeanSquaredError())

# Fit model
net.fit(xtf, utf, epochs=epochs, shuffle=True)  # , callbacks=[early_stop])  # , batch_size=nodes)

# with tf.GradientTape() as tape:
#     tape.watch(xtf)
#     out = net.predict(xtf, batch_size=nodes)
#
# out_x = tape.gradient(out, xtf)
# out_xx = tape.gradient(out_x, xtf)
# rhs = -alpha * out_x + out_xx
# print(rhs.shape)
# print(irk.gl_weights.shape)
# quit()
# u_next = out[:, -1] + dt * irk.gl_weights

out = net.predict(xtf, batch_size=nodes)
# print(out.shape)
# quit()

# Compute RK stage vector, rhs = du/dt
rhs = -alpha * out[1, :, :] + out[2, :, :]
u0_pred = out[0, :, :] - dt * tf.matmul(rhs, IRK.rk_matrix_tf32)

# u1_true = solution_dirichlet(x0, dt, alpha)
u1_true = solution_periodic(x0, dt, alpha)

plt.figure()
plt.plot(x0, out[0, :, -1], '--', label='Net solution')
plt.plot(x0, u0, label='Initial condition')
plt.plot(x0, u1_true, label='True solution')
for i in range(IRK.order):
    plt.plot(x0, out[0, :, i], '--', label='stage ' + str(i))
plt.title('Next stage prediction: solution')
plt.legend(loc='best')
plt.grid(True)

# plt.figure()
# plt.plot(x0, out[1, :, -1], '--', label='u_x')
# plt.plot(x0, out[2, :, -1], '--', label='u_xx')
# plt.title('Next stage prediction: derivatives')
# plt.legend(loc='best')
# plt.grid(True)

plt.figure()
plt.plot(x0, u0_pred[:, :], '--', label='u0 prediction')
plt.plot(x0, u0, 'o--', label='u0')
plt.legend(loc='best')
plt.grid(True)

plt.show()
