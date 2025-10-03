import irk_coefficients as irk
import matplotlib.pyplot as plt
import neural_net as nn
import newton as newton
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import x_grid as grid

keras.backend.clear_session()

# Part One: Lorenz system
# Initialize IRK coefficient matrix...
order = 100 # 100  # 100  # 50  # 10
# nodes = 32
IRK = irk.IRK(order=order)
IRK.build_matrix()

dt = 0.8
# Newton solver configuration
use_adaptive = True  # Set to False to use original Newton method
adaptive_method = 'krylov'  # Options: 'hybr', 'lm', 'broyden1', 'df-sane', 'krylov'

# Net parameters
alpha = 1.0
parameters = [dt, alpha]
# neurons = nodes
activation = "elu"  # 'tanh'
optimizer = "adam"
epochs = 600 # 300 # 600 # 10 # 400 # 1500

# Transfer learning configuration
use_transfer_learning = False  # Disabled in favor of Euler stepping each timestep
transfer_epochs_reduction = 0.5  # Reduce epochs by this factor after first step

# Neural network initialization configuration
use_euler_guess = True  # Use explicit Euler to initialize neural network at EVERY timestep

# Main net loop for Lorenz system
steps = 3  # Increased to demonstrate transfer learning
q = np.array([10.54, 4.112, 35.82])
# q = np.array([1.0, -1.0, -1.0])
q_loop = np.zeros((3, order + 1, steps))
previous_net = None  # Store previous network for transfer learning

for idx in range(steps):
    # Lorenz with IRK:
    u0 = np.array([q])  # , q1, q2])

    # Generate initial guess using Euler if enabled
    if use_euler_guess:
        # Use sequential Euler steps from node to node for better accuracy
        # This follows the actual trajectory more closely than single jumps
        euler_guess = np.zeros((3, order + 1))

        # Start from initial condition
        current_state = u0[0, :].copy()
        current_time = 0.0

        # Get IRK nodes and transform from [-1,1] to [0,1]
        if hasattr(IRK, 'nodes'):
            # Transform Gauss-Legendre nodes from [-1,1] to [0,1]
            irk_nodes_raw = IRK.nodes
            irk_nodes = 0.5 * (irk_nodes_raw + 1.0)  # Map [-1,1] -> [0,1]
        else:
            irk_nodes = np.linspace(0, 1, order, endpoint=False)

        # Sort nodes to ensure we step in time order
        irk_nodes = np.sort(irk_nodes)

        print(f"Raw IRK nodes [-1,1]: {irk_nodes_raw[:5]}..." if len(irk_nodes_raw) > 5 else f"Raw IRK nodes [-1,1]: {irk_nodes_raw}")
        print(f"Transformed nodes [0,1]: {irk_nodes[:5]}..." if len(irk_nodes) > 5 else f"Transformed nodes [0,1]: {irk_nodes}")

        # Step through each IRK node sequentially
        for i in range(order):
            target_time = irk_nodes[i]
            step_dt = (target_time - current_time) * dt

            # Take Euler step to reach this node time
            if step_dt > 0:  # Only step forward if we need to
                rhs, _ = newton.lorenz_dynamics(current_state[None, :])
                current_state = current_state + step_dt * rhs[0, :]
                current_time = target_time

            euler_guess[:, i] = current_state

        # Final solution: step from last node to t=1.0
        final_step_dt = (1.0 - current_time) * dt
        if final_step_dt > 0:
            rhs, _ = newton.lorenz_dynamics(current_state[None, :])
            final_state = current_state + final_step_dt * rhs[0, :]
        else:
            final_state = current_state

        euler_guess[:, -1] = final_state

        print(f"Sequential Euler trajectory:")
        print(f"  Initial state: {u0[0, :]}")
        print(f"  First node: {euler_guess[:, 0]}")
        print(f"  Final solution: {euler_guess[:, -1]}")

        # Manually calculate what the loss should be for this Euler guess
        print("\nManual loss calculation for Euler guess:")
        # Simulate what the custom_loss function does
        euler_tensor = tf.constant(euler_guess[np.newaxis, :, :], dtype=tf.float32)

        # Apply the same logic as custom_loss
        sigma, beta, rho = 10.0, 8.0/3.0, 28.0
        u1 = euler_tensor
        rhs = tf.stack([
            sigma * (u1[:, 1, :] - u1[:, 0, :]),
            u1[:, 0, :] * (rho - u1[:, 2, :]) - u1[:, 1, :],
            u1[:, 0, :] * u1[:, 1, :] - beta * u1[:, 2, :]
        ])

        # Convert IRK matrix to tensor if needed
        if hasattr(IRK, 'rk_matrix_tf32'):
            A_matrix = IRK.rk_matrix_tf32
        else:
            A_matrix = tf.constant(IRK.rk_matrix, dtype=tf.float32)

        # Backward IRK step
        u0_reconstructed = u1 - dt * tf.transpose(tf.matmul(rhs, A_matrix), perm=(1, 0, 2))

        # Error vs true initial condition
        error = tf.cast(u0[:, :, None], tf.float32) - u0_reconstructed
        manual_loss = tf.reduce_mean(tf.square(error))

        print(f"  Expected loss for Euler guess: {manual_loss.numpy():.6f}")
        print(f"  True u0: {u0[0, :]}")
        print(f"  Reconstructed u0: {u0_reconstructed[0, :, 0].numpy()}")
    else:
        euler_guess = None

    # Make neural net
    utf = tf.reshape(tf.convert_to_tensor(u0), (u0.shape[0], u0.shape[1]))
    net = nn.NeuralNet_LorenzStepper(
        parameters=parameters, irk=IRK, neurons=2*q.shape[0], activation=activation
    )
    loss_fn = net.custom_loss
    net.compile(
        optimizer=optimizer, loss=lambda y_true, y_pred: loss_fn(y_true, y_pred)
    )

    # Build the network by calling it once
    initial_output = net(utf)

    # Check what the network outputs before any training
    print(f"\nNetwork output before any training:")
    print(f"  Shape: {initial_output.shape}")
    print(f"  First few values: {initial_output[0, :, :3].numpy()}")

    # Calculate initial loss with random weights
    initial_loss_random = loss_fn(utf, initial_output)
    print(f"  Initial loss (random weights): {initial_loss_random:.6f}")

    # Use Euler guess to guide training if enabled
    if use_euler_guess and euler_guess is not None:
        print("\nTraining will use Euler-guided initialization...")

        # Reshape Euler guess to match network output format: (1, 3, order+1)
        euler_target = euler_guess[np.newaxis, :, :]  # Shape: (1, 3, order+1)

        print(f"Euler target shape: {euler_target.shape}")
        print(f"Euler target first few values: {euler_target[0, :, :3]}")

        # Directly initialize network weights to output Euler guess
        print("Directly initializing network weights with Euler guess...")

        # Get current weights
        weights = net.get_weights()

        # Set the final layer bias to output the Euler guess
        # The final layer should have shape (3 * (order + 1),) for the bias
        euler_flat = euler_guess.flatten()  # Flatten: (3, 33) -> (99,)

        if len(weights) >= 2:  # Make sure we have bias weights
            final_bias = weights[-1]  # Last element should be final layer bias
            print(f"Final bias shape: {final_bias.shape}")
            print(f"Euler flat shape: {euler_flat.shape}")

            if final_bias.shape == euler_flat.shape:
                # Set bias to output Euler guess directly
                weights[-1] = euler_flat
                net.set_weights(weights)
                print("Successfully set final layer bias to Euler values")

                # Verify the initialization worked
                initialized_output = net(utf)
                print(f"\nNetwork output after direct initialization:")
                print(f"  First few values: {initialized_output[0, :, :3].numpy()}")
                print(f"  Should match Euler: {euler_guess[:, :3]}")

                # Check initial loss after direct initialization
                initial_loss = loss_fn(utf, initialized_output)
                print(f"  Initial loss after direct initialization: {initial_loss:.6f}")
            else:
                print(f"Shape mismatch: bias {final_bias.shape} vs Euler {euler_flat.shape}")
        else:
            print("Could not find bias weights to modify")

    # Set training epochs based on initialization method
    current_epochs = epochs
    if use_euler_guess and euler_guess is not None:
        # Reduce epochs since we start with physics-based initialization
        current_epochs = int(epochs * 0.8)  # 20% reduction since we start closer
        print(f"Reduced epochs from {epochs} to {current_epochs} due to Euler initialization")

        # For subsequent timesteps, reduce even further since Euler should be more accurate
        if idx > 0:
            current_epochs = int(epochs * 0.6)  # 40% reduction for later timesteps
            print(f"Further reduced to {current_epochs} epochs for timestep {idx} (improved Euler accuracy)")

    # Fit model
    net.fit(
        utf, utf, epochs=current_epochs, shuffle=True
    )  # , callbacks=[early_stop])  # , batch_size=nodes)

    # Store current network weights for next timestep transfer
    if use_transfer_learning:
        previous_net = net.get_weights()  # Just store the weights

    out = net.predict(utf)  # , batch_size=nodes)

    # sigma = 10
    # beta = 8 / 3
    # rho = 28
    # rhs = tf.convert_to_tensor([sigma * (out[:, 1, :] - out[:, 0, :]),
    #                         out[:, 0, :] * (rho - out[:, 2, :]) - out[:, 1, :],
    #                         out[:, 0, :] * out[:, 1, :] - beta * out[:, 2, :]])
    # u0_out = np.asarray(out - dt * tf.transpose(tf.matmul(rhs, IRK.rk_matrix_tf32), perm=(1, 0, 2)))

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(u0_out[:, 0, :].flatten(), u0_out[:, 1, :].flatten(), u0_out[:, 2, :].flatten(), label='predicted u0')
    # ax.scatter(u0[:, 0], u0[:, 1], u0[:, 2])
    # ax.scatter(out[:, 0, :].flatten(), out[:, 1, :].flatten(), out[:, 2, :].flatten(), label='predicted rk stages')
    # plt.show()

    # Use output as guess for Newton solver
    nt = 2
    qs = np.zeros((3, order + 1, nt))
    for i in range(order + 1):
        qs[:, i, 0] = q

    for i in range(1, nt):
        # Newton iteration for rhs evaluations
        if use_adaptive:
            k_vec = newton.newton_irk_adaptive(
                q, dt=dt, irk=IRK, threshold=1.0e-10, max_iterations=50000,
                guess=out[0, :, :], method=adaptive_method
            )
        else:
            k_vec = newton.newton_irk(
                q, dt=dt, irk=IRK, threshold=1.0e-10, max_iterations=50000, guess=out[0, :, :]
            )
        # GL stages
        qs[:, :-1, i] = q[:, None] + dt * np.transpose(
            np.tensordot(IRK.rk_matrix, k_vec, axes=([1], [0])), axes=([1, 0])
        )
        # Update
        q += (
            0.5 * dt * np.tensordot(IRK.weights, k_vec, axes=([0], [0]))
        )  # 0.5 * dt * (k1 + k2)
        qs[:, -1, i] = q

    q_loop[:, :, idx] = qs[:, :, -1]

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.set_title(
    "Lorenz system implicit RK advance, dt=%.3e" % dt + ", err threshold 1.0e-10"
)
for i in range(steps):
    ax.scatter(
        q_loop[0, :, i],
        q_loop[1, :, i],
        q_loop[2, :, i],
        label="stages of time-step " + str(i),
    )
# ax.scatter(out[:, 0, :].flatten(), out[:, 1, :].flatten(), out[:, 2, :].flatten(), label='predicted rk stages')
# ax.scatter(qs[0, :, :].flatten(), qs[1, :, :].flatten(), qs[2, :, :].flatten(), label='newton iterated rk stages')
# plt.legend(loc='best')
# for i in range(nt):
#     ax.plot(qs[0, :, i], qs[1, :, i], qs[2, :, i], 'o--')
# print(qs)
plt.show()

quit()


# Part Two: Advection-Diffusion with net:
def solution_dirichlet(x, t, a):
    # Problem to solve... first mode of dirichlet linear advection-diffusion
    return (
        np.exp(a * 0.5 * (x - a * 0.5 * t))
        * np.sin(np.pi * x)
        * np.exp(-(np.pi**2.0) * t)
    )


def solution_periodic(x, t, a):
    # Problem to solve... periodic traveling mode of linear advection-diffusion
    return np.sin(2.0 * np.pi * (x - a * t)) * np.exp(-((2.0 * np.pi) ** 2.0) * t)


# Net parameters
dt = 0.05
alpha = 1.0
parameters = [dt, alpha]
# neurons = nodes
activation = "tanh"
optimizer = "adam"
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
plt.plot(x0, u0, "o--", label="Initial condition")
plt.legend(loc="best")
plt.grid(True)
plt.show()

# Boundary points
lb = 0.0
rb = 1.0
boundary = np.array([lb, rb])

# Make neural net
xtf = tf.reshape(tf.convert_to_tensor(x0), (nodes, 1))
utf = tf.reshape(tf.convert_to_tensor(u0), (nodes, 1))
net = nn.NeuralNet_AdvectionDiffusion(
    x=x0,
    u=u0,
    bc=boundary,
    parameters=parameters,
    irk=IRK,
    neurons=nodes,
    activation=activation,
)
loss_fn_ad = net.loss_with_bc
net.compile(optimizer=optimizer, loss=lambda y_true, y_pred: loss_fn_ad(y_true, y_pred))

# Fit model
net.fit(xtf, utf, epochs=epochs, shuffle=True)
out = net.predict(xtf, batch_size=nodes)

# Compute RK stage vector, rhs = du/dt
rhs = -alpha * out[1, :, :] + out[2, :, :]
u0_pred = out[0, :, :] - dt * tf.matmul(rhs, IRK.rk_matrix_tf32)

u1_true = solution_periodic(x0, dt, alpha)

plt.figure()
plt.plot(x0, out[0, :, -1], "--", label="Net solution")
plt.plot(x0, u0, label="Initial condition")
plt.plot(x0, u1_true, label="True solution")
for i in range(IRK.order):
    plt.plot(x0, out[0, :, i], "--", label="stage " + str(i))
plt.title("Next stage prediction: solution")
plt.legend(loc="best")
plt.grid(True)

plt.figure()
plt.plot(x0, u0_pred[:, :], "--", label="u0 prediction")
plt.plot(x0, u0, "o--", label="u0")
plt.legend(loc="best")
plt.grid(True)

plt.show()
