import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Physical and numerical parameters
L_A, L_B = 1.0, 1.0
Nx_A, Nx_B = 50, 50
alpha_A, alpha_B = 0.01, 0.01
beta = 0.1  # nonlinearity factor
dx = L_A / (Nx_A - 1)
dt = 0.4 * dx**2 / max(alpha_A, alpha_B)
Tmax = 100
Nt = int(Tmax / dt)

# Snapshots
num_snapshots = 10
snapshot_interval = Nt // (num_snapshots - 1)

# Initial conditions: piecewise linear
T_A = np.linspace(20.0, 20.0, Nx_A)
T_B = np.linspace(10.0, 10.0, Nx_B)
EXACT_SOLUTION = 15.0  # Exact solution for testing
# Storage
history = []

def l2_error(T1, T2):
    """Compute L2 error between two temperature profiles."""
    return np.sqrt(np.sum((T1 - T2)**2) / len(T1))


# Classic heat equation update (explicit, Neumann BCs)
def update_heat_explicit(T, alpha, dx, dt, n):
    T_new = T.copy()
    for i in range(1, len(T) - 1):
        T_new[i] = T[i] + alpha * dt / dx**2 * (T[i-1] - 2*T[i] + T[i+1])
    T_new[0] = T_new[1]       # Neumann BC left
    T_new[-1] = T_new[-2]     # Neumann BC right
    
    # Add nonlinear source term
    T_new += beta * dt * np.exp(-n*dt)
    return T_new

# Fixed-point iteration parameters
n_solves = 0
max_iter = 100
tol = 1e-6
iter_conv = -1
cpl_frequency = 100  # Frequency of coupling iterations
# Time stepping loop
for n in range(0, Nt):

    if n % cpl_frequency == 0:
        T_A_lag = T_A.copy()
        T_B_lag = T_B.copy()

    # Save snapshot
    if n % snapshot_interval == 0:
        history.append(np.concatenate((T_A, T_B)))


    T_A_guess = T_A_lag.copy()
    T_B_guess = T_B_lag.copy()

    for iter_count in range(1, max_iter + 1):

        # Interface
        T_if = 0.5 * (T_A_guess[-1] + T_B_guess[0])

        T_A_guess[-1] = T_if
        T_B_guess[0] = T_if

        T_A_guess = update_heat_explicit(T_A_guess, alpha_A, dx, dt, n)
        T_B_guess = update_heat_explicit(T_B_guess, alpha_B, dx, dt, n)
        n_solves += 1

        err_a = np.abs(T_A_guess[-1] - T_if)
        err_b = np.abs(T_B_guess[0] - T_if)

        if err_a < tol and err_b < tol:
            break 

    global_err = l2_error(T_A_guess, EXACT_SOLUTION) + l2_error(T_B_guess, EXACT_SOLUTION)

    if (global_err < tol and iter_conv == -1):
        iter_conv = n
        print(f"Converged at time step {n} after {n_solves} iterations with global error {global_err:.6f}")

    # Print iteration info at each snapshot
    if n_solves % snapshot_interval == 0 or n == Nt - 1:
        print(f"Time step {n}: Fixed-point iterations = {iter_count}, global_err= {global_err:.6f}")

    # Update for next step
    T_A = T_A_guess
    T_B = T_B_guess

history.append(np.concatenate((T_A, T_B)))


# Plotting with interactive slider
x_A = np.linspace(0, L_A, Nx_A)
x_B = np.linspace(L_A, L_A + L_B, Nx_B)
x_full = np.concatenate((x_A, x_B))

fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(bottom=0.2)

line, = ax.plot(x_full, history[0], label='t=0.00')
ax.set_xlabel("x")
ax.set_ylabel("Temperature")
ax.set_title("Nonlinear Explicit Coupling at Interface")
ax.grid(True)
legend = ax.legend()

# Slider axis
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(ax_slider, 'Time', 0, len(history)-1, valinit=0, valstep=1)

def update(val):
    idx = int(slider.val)
    line.set_ydata(history[idx])
    time_val = idx * snapshot_interval * dt
    line.set_label(f't={time_val:.2f}')
    ax.legend()
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.show()

