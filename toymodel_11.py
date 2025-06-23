import numpy as np
import matplotlib.pyplot as plt

# Physical and numerical parameters
L_A, L_B = 1.0, 1.0
Nx_A, Nx_B = 50, 50
alpha_A, alpha_B = 0.01, 0.01
beta = 0.01  # nonlinearity factor
dx = L_A / (Nx_A - 1)
dt = 0.4 * dx**2 / max(alpha_A, alpha_B)
Tmax = 10
Nt = int(Tmax / dt)

# Snapshots
num_snapshots = 10
snapshot_interval = Nt // (num_snapshots - 1)

# Initial conditions: piecewise linear
T_A = np.linspace(10.0, 20.0, Nx_A)
T_B = np.linspace(10.0, 10.0, Nx_B)

# Storage
history = []

# Nonlinear thermal diffusivity function
def k(T, alpha, beta):
    return alpha * (1 + beta * T)

# Nonlinear heat equation update (explicit, Neumann BCs)
def update_heat_nonlinear_explicit(T, alpha, beta, dx, dt, t, gamma=0.01, omega=0.1):
    T_new = T.copy()
    for i in range(1, len(T) - 1):
        km = 0.5 * (k(T[i], alpha, beta) + k(T[i-1], alpha, beta))
        kp = 0.5 * (k(T[i], alpha, beta) + k(T[i+1], alpha, beta))
        # Nonlinear source term in time
        source = gamma * T[i] * np.sin(omega * t)
        T_new[i] = T[i] + dt / dx**2 * (km * (T[i-1] - T[i]) + kp * (T[i+1] - T[i])) + dt * source
    T_new[0] = T_new[1]       # Neumann BC left
    T_new[-1] = T_new[-2]     # Neumann BC right
    return T_new

# Fixed-point iteration parameters
max_iter = 3
tol = 1e-6

# Time stepping loop
for n in range(0, Nt):

    # Save snapshot
    if n % snapshot_interval == 0:
        history.append(np.concatenate((T_A, T_B)))


    T_A_guess = T_A.copy()
    T_B_guess = T_B.copy()

    for iter_count in range(1, max_iter + 1):

        # Interface
        T_if = 0.5 * (T_A_guess[-1] + T_B_guess[0])

        T_A_guess[-1] = T_if
        T_B_guess[0] = T_if

        T_A_guess = update_heat_nonlinear_explicit(T_A_guess, alpha_A, beta, dx, dt, n * dt)
        T_B_guess = update_heat_nonlinear_explicit(T_B_guess, alpha_B, beta, dx, dt, n * dt)

        err_a = np.abs(T_A_guess[-1] - T_if)
        err_b = np.abs(T_B_guess[0] - T_if)

        if err_a < tol and err_b < tol:
            break 
            
    # Print iteration info at each snapshot
    if n % snapshot_interval == 0 or n == Nt - 1:
        print(f"Time step {n}: Fixed-point iterations = {iter_count}, err_a= {err_a:.6f}")

    # Update for next step
    T_A = T_A_guess
    T_B = T_B_guess

history.append(np.concatenate((T_A, T_B)))


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

