import numpy as np
import matplotlib.pyplot as plt

# Physical and numerical parameters
L_A, L_B = 1.0, 1.0
Nx_A, Nx_B = 50, 50
alpha_A, alpha_B = 0.01, 0.01
beta = 0.01  # nonlinearity factor
dx = L_A / (Nx_A - 1)
dt = 0.4 * dx**2 / max(alpha_A, alpha_B)
Tmax = 10000
Nt = int(Tmax / dt)

# Snapshots
num_snapshots = 20
snapshot_interval = Nt // (num_snapshots - 1)
cpl_freq = 1

# Initial conditions: piecewise linear
T_A = np.linspace(20.0, 20.0, Nx_A)
T_B = np.linspace(10.0, 10.0, Nx_B)

# Storage
history = []

def l2_error(T1, T2):
    """Compute L2 error between two temperature profiles."""
    return np.sqrt(np.sum((T1 - T2)**2) / len(T1))

def linf_error(T, T_ref):
    return np.max(np.abs(T - T_ref))

# Classic heat equation update (explicit, Neumann BCs)
def update_heat_explicit(T, alpha, dx, dt, t, beta):
    T_new = T.copy()
    source = beta * np.sin(t)
    for i in range(1, len(T) - 1):
        T_new[i] = T[i] + alpha * dt / dx**2 * (T[i-1] - 2*T[i] + T[i+1]) + dt * source
    T_new[0] = T_new[1]       # Neumann BC left
    T_new[-1] = T_new[-2]     # Neumann BC right
    return T_new

def T_exact(t, T0, beta):
    return T0 + beta * (1 - np.cos(t))

T0 = 15.0  # For comparison with exact solution

history.append(np.concatenate((T_A, T_B)))
n_solves = 0
tol = 1e-6
iter_conv = -1
for n in range(0, Nt):
    t = n * dt

    if (n%cpl_freq == 0):
        T_A_int = T_A[-1]
        T_B_int = T_B[0]

    T_interface = 0.5 * (T_A_int + T_B_int)
    T_A_new = T_A.copy()
    T_B_new = T_B.copy()
    T_A_new[-1] = T_interface
    T_B_new[0] = T_interface

    # Update both domains with source term
    T_A_new = update_heat_explicit(T_A_new, alpha_A, dx, dt, t, beta)
    T_B_new = update_heat_explicit(T_B_new, alpha_B, dx, dt, t, beta)
    n_solves += 1

    T_A = T_A_new
    T_B = T_B_new

    exact_val = T_exact(t, T0, beta)
    global_err = linf_error(T_A_new, exact_val) + linf_error(T_B_new, exact_val)

    # Print iteration info at each snapshot
    if n % snapshot_interval == 0 or n == Nt - 1:
        print(f"Time step {n}: global_err= {global_err:.6f}")

    # Save snapshot
    if n % snapshot_interval == 0 or n == Nt-1:
        print(f"Time step {n}")
        history.append(np.concatenate((T_A, T_B)))

    if (global_err < tol and iter_conv == -1):
        iter_conv = n
        print(f"Converged at time step {n} after {n_solves} iterations with global error {global_err:.6f}")
        break


# Plotting
x_A = np.linspace(0, L_A, Nx_A)
x_B = np.linspace(L_A, L_A + L_B, Nx_B)
x_full = np.concatenate((x_A, x_B))

plt.figure(figsize=(10, 6))
for i, T in enumerate(history):
    plt.plot(x_full, T, label=f't={i * snapshot_interval * dt:.2f}')
plt.xlabel("x")
plt.ylabel("Temperature")
plt.title("Nonlinear Explicit Coupling at Interface")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("output.png", dpi=300)
print("Saved to output.png")
