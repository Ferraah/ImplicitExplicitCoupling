import numpy as np
import matplotlib.pyplot as plt

# Parameters
L_A, L_B = 1.0, 1.0       # lengths of domain A and B
Nx_A, Nx_B = 50, 50       # number of spatial points
alpha_A, alpha_B = 0.1, 0.1  # thermal diffusivity
dx = L_A / (Nx_A - 1)
dt = 0.4 * dx**2 / max(alpha_A, alpha_B)  # stability for explicit scheme
Tmax = 1.0                 # total simulation time
Nt = int(Tmax / dt)
coupling_frequency = 10# exchange interface data every N steps
snapshots = Nt//10
print(Nt, coupling_frequency)

# Initial conditions
T_A = np.zeros(Nx_A)
T_B = np.zeros(Nx_B)
# T_A[int(Nx_A / 4):int(Nx_A / 2)] = 1.0  # initial heat pulse in A
# T_B[int(Nx_B / 2):int(3 * Nx_B / 4)] = 1.0  # initial heat pulse in B
T_A[:] = 30.0 
T_B[:] = 10.0

# For visualization
history = []

def update_heat(T, alpha, dx, dt):
    T_new = T.copy()
    T_new[1:-1] = T[1:-1] + alpha * dt / dx**2 * (T[2:] - 2 * T[1:-1] + T[:-2])
    return T_new

for n in range(Nt):
    # Update each domain separately
    T_A_new = update_heat(T_A, alpha_A, dx, dt)
    T_B_new = update_heat(T_B, alpha_B, dx, dt)

    # Coupling: exchange boundary at interface every 'coupling_frequency' steps
    if n % coupling_frequency == 0:
        T_interface = 0.5 * (T_A[-1] + T_B[0])
        T_A_new[-1] = T_interface
        T_B_new[0]  = T_interface
    else:
        # Neumann (no-flux) condition if not coupled
        T_A_new[-1] = T_A[-2]
        T_B_new[0] = T_B[1]

    T_A, T_B = T_A_new, T_B_new

    # Save for plotting
    if n % snapshots == 0:
        full_T = np.concatenate((T_A, T_B))
        history.append(full_T)

# Plot results
x_A = np.linspace(0, L_A, Nx_A)
x_B = np.linspace(L_A, L_A + L_B, Nx_B)
x_full = np.concatenate((x_A, x_B))

plt.figure(figsize=(10, 6))
for i, T in enumerate(history):
    plt.plot(x_full, T, label=f't={i*50*dt:.2f}')
plt.xlabel('x')
plt.ylabel('Temperature')
plt.title('Coupled Heat Equation with Intermittent Interface Exchange')
plt.legend()
plt.grid()
plt.savefig("output.png")

