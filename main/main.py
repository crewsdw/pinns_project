import numpy as np
import irk_coefficients as irk
import neural_net as nn
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import x_grid as grid

keras.backend.clear_session()


# Problem to solve... first mode of linear advection-diffusion
def solution_dirichlet(x, t, a):
    return np.exp(a * 0.5 * (x - a * 0.5 * t)) * np.sin(np.pi * x) * np.exp(-(np.pi ** 2.0) * t)


def solution_periodic(x, t, a):
    return np.sin(2.0 * np.pi * (x - a * t)) * np.exp(-(2.0 * np.pi) ** 2.0 * t)


# Initialize IRK coefficient matrix...
order = 10  # 4
nodes = 32
IRK = irk.IRK(order=order)
IRK.build_matrix()

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
net = nn.NeuralNet(x=x0, u=u0, bc=boundary, parameters=parameters,
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
