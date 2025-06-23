import numpy as np
import matplotlib.pyplot as plt

# Physical and numerical parameters
L_A, L_B = 1.0, 1.0
Nx_A, Nx_B = 50, 50
alpha_A, alpha_B = 0.01, 0.005
dx = L_A / (Nx_A - 1)
dt = 0.4 * dx**2 / max(alpha_A, alpha_B)
Tmax = 4.0
Nt = int(Tmax / dt)

# Coupling parameters
max_iter = 50
tolerance = 1e-8

# Snapshots to plot
num_snapshots = 6
snapshot_interval = Nt // (num_snapshots - 1)

# Initial conditions
# T_A = np.ones(Nx_A) * 10.0
# T_B = np.ones(Nx_B) * 30.0
# T_A = np.linspace(10.0, 20.0, Nx_A)
# T_B = np.linspace(20, 10.0, Nx_B)

T_A = np.linspace(10.0, 20.0, Nx_A)
T_B = np.linspace(20.0, 10.0, Nx_B)

# Storage
history = []

# Heat equation (explicit, with Neumann BCs)
def update_heat_explicit(T, alpha, dx, dt):
    T_new = T.copy()
    T_new[1:-1] = T[1:-1] + alpha * dt / dx**2 * (T[2:] - 2*T[1:-1] + T[:-2])
    # Neumann BCs: insulated ends
    T_new[0] = T_new[1]
    T_new[-1] = T_new[-2]
    return T_new

for n in range(Nt):
    T_A_guess = T_A.copy()
    T_B_guess = T_B.copy()

    n_it = 0
    for _ in range(max_iter):
        n_it += 1
        T_A_new = update_heat_explicit(T_A_guess, alpha_A, dx, dt)
        T_B_new = update_heat_explicit(T_B_guess, alpha_B, dx, dt)

        # Impose flux continuity and equal value at interface
        # Let T_if be the value at the interface
        # From flux continuity:
        num = alpha_A * T_A_new[-2] + alpha_B * T_B_new[1]
        denom = alpha_A + alpha_B
        T_if = num / denom

        # Apply interface values
        T_A_new[-1] = T_if
        T_B_new[0]  = T_if

        # Check convergence
        if np.abs(T_if - T_A_guess[-1]) < tolerance and np.abs(T_if - T_B_guess[0]) < tolerance:
            break

        T_A_guess = T_A_new
        T_B_guess = T_B_new

    print(n, n_it)
    T_A = T_A_new
    T_B = T_B_new

    # Save snapshots
    if n % snapshot_interval == 0 or n == Nt - 1:
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
plt.title("Implicit Coupling with Flux Continuity")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("output.png", dpi=300)
print("Saved to output.png")
