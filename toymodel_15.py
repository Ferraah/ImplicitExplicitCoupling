import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 1.0
Nx = 101
alpha = 0.01
dx = L / (Nx - 1)
dt = 0.001
T = 50
Nt = int(T / dt)
x = np.linspace(0, L, Nx)

# Domain splitting
xI = 0.5
iI = np.argmin(np.abs(x - xI))

# Initial condition
def initial_condition(x):
    return np.sin(np.pi * x)

# Exact solution for reference
def exact_solution(x, T, alpha):
    return np.sin(np.pi * x) * np.exp(-np.pi**2 * alpha * T)

# Subdomain solver (black-box style)
def solve_subdomain(u, left_bc, right_bc, alpha, dx, dt, steps):
    u = u.copy()
    for _ in range(steps):
        u[0] = left_bc
        u[-1] = right_bc
        u[1:-1] += alpha * dt / dx**2 * (u[2:] - 2*u[1:-1] + u[:-2])
    return u

# Monolithic solver
def monolithic_solver(u0):
    u = u0.copy()
    for _ in range(Nt):
        u[0] = 0.0
        u[-1] = 0.0
        u[1:-1] += alpha * dt / dx**2 * (u[2:] - 2*u[1:-1] + u[:-2])
    return u

# Explicit coupling
def explicit_coupling(u0):
    u1 = u0[:iI+1].copy()
    u2 = u0[iI:].copy()
    for n in range(Nt):
        u1[0] = 0.0
        u1[-1] = u2[0]  # use old value from u2
        u1[1:-1] += alpha * dt / dx**2 * (u1[2:] - 2*u1[1:-1] + u1[:-2])

        u2[-1] = 0.0
        u2[0] = u1[-1]  # use just updated value from u1
        u2[1:-1] += alpha * dt / dx**2 * (u2[2:] - 2*u2[1:-1] + u2[:-2])
    return np.concatenate((u1[:-1], u2))

# Schwarz iteration coupling
def schwarz_coupling(u0, max_iter=10, tol=1e-4, n_overlap=10):
    # Define subdomain indices with overlap
    left_end = iI + n_overlap
    right_start = iI - n_overlap

    u1 = u0[:left_end + 1].copy()
    u2 = u0[right_start:].copy()
    history = [np.concatenate((u1[:iI], u2))]  # for time evolution

    for n in range(Nt):
        u1_k = u1.copy()
        u2_k = u2.copy()
        for _ in range(max_iter):
            # Dirichlet BCs at physical boundaries
            u1_k[0] = 0.0
            u2_k[-1] = 0.0

            # Exchange overlap: set right overlap of u1 from u2, and left overlap of u2 from u1
            u1_k[-n_overlap:] = u2_k[:n_overlap]
            u2_k[:n_overlap] = u1_k[-n_overlap:]

            # Update interior points
            u1_k[1:-1] += alpha * dt / dx**2 * (u1_k[2:] - 2*u1_k[1:-1] + u1_k[:-2])
            u2_k[1:-1] += alpha * dt / dx**2 * (u2_k[2:] - 2*u2_k[1:-1] + u2_k[:-2])

            # Convergence check on overlap
            if np.linalg.norm(u1_k[-n_overlap:] - u2_k[:n_overlap], ord=np.inf) < tol:
                break

        u1 = u1_k
        u2 = u2_k
        # Store for time evolution (avoid double-counting overlap)
        history.append(np.concatenate((u1[:iI], u2)))

    return np.array(history)

# Run all solvers
u0 = initial_condition(x)
u_monolithic = monolithic_solver(u0)
u_explicit = explicit_coupling(u0)
u_schwarz = schwarz_coupling(u0)
u_exact = exact_solution(x, T, alpha)

# Plot comparison
plt.figure(figsize=(10, 5))
plt.plot(x, u_exact, 'k--', label='Exact')
plt.plot(x, u_monolithic, label='Monolithic')
plt.plot(x, u_explicit, label='Explicit Coupling')
plt.plot(x, u_schwarz[-1], label='Schwarz Coupling')  # Use the last time step
plt.xlabel('x')
plt.ylabel('u(x,T)')
plt.title('Comparison of Coupling Strategies for 1D Heat Equation')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("output.png")
