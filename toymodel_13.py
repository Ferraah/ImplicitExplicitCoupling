import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Parameters
L = 1.0           # domain length
Nx = 100          # total number of spatial points
dx = L / (Nx - 1)
x = np.linspace(0, L, Nx)

# Split domain
mid = Nx // 2
x1 = x[:mid+1]    # left domain including interface
x2 = x[mid:]      # right domain including interface

# Diffusivities
D1 = 1.0
D2 = 0.1

# Time parameters
T = 0.5           # total time
dt = 0.001
Nt = int(T/dt)

# Initial condition: temperature jump at interface
u_init = np.zeros(Nx)
u_init[:mid+1] = 1.0   # left side initially 1
u_init[mid+1:] = 0.0   # right side initially 0

def build_matrix(N, D, dx, dt):
    """
    Build the implicit Euler matrix for 1D heat equation with Dirichlet BC.
    """
    r = D * dt / dx**2
    A = np.eye(N) * (1 + 2*r)
    for i in range(N-1):
        A[i, i+1] = -r
        A[i+1, i] = -r
    # Dirichlet BCs on boundaries: modify first and last rows
    A[0, :] = 0
    A[0, 0] = 1
    A[-1, :] = 0
    A[-1, -1] = 1
    return A

def implicit_step(u_old, A, b):
    """
    Solve A u_new = b
    """
    from numpy.linalg import solve
    return solve(A, b)

def explicit_coupling(u_explicit, A1, A2, dt, dx, mid):
    """
    One explicit coupling time step:
    Conserve energy at the interface using flux and temperature continuity.
    """
    u1_old = u_explicit[:mid+1]
    u2_old = u_explicit[mid:]
    
    b1 = u1_old.copy()
    b2 = u2_old.copy()
    b1[0] = 1.0
    b2[-1] = 0.0

    # --- Interface coupling ---
    # 1. Flux continuity: D1*(u1[-1]-u1[-2])/dx = D2*(u2[1]-u2[0])/dx
    #    => u1[-1] = u1[-2] + (D2/D1)*(u2[1]-u2[0])
    # 2. Temperature continuity: u2[0] = u1[-1]
    # Impose these as linear constraints

    # Modify last row of A1 and b1 for interface
    A1[-1, :] = 0
    A1[-1, -1] = 1
    A1[-1, -2] = -1
    b1[-1] = (D2/D1) * (u2_old[1] - u2_old[0])

    # Modify first row of A2 and b2 for interface
    A2[0, :] = 0
    A2[0, 0] = 1
    b2[0] = u1_old[-1]

    u1_new = implicit_step(u1_old, A1, b1)
    u2_new = implicit_step(u2_old, A2, b2)
    u_new = np.concatenate((u1_new[:-1], u2_new))
    return u_new

def implicit_coupling(u_implicit, A1, A2, dt, dx, mid, tol=1e-6, max_iter=20):
    """
    One implicit coupling time step:
    Conserve energy at the interface using flux and temperature continuity.
    """
    u1_old = u_implicit[:mid+1]
    u2_old = u_implicit[mid:]
    
    for _ in range(max_iter):
        b1 = u1_old.copy()
        b2 = u2_old.copy()
        b1[0] = 1.0
        b2[-1] = 0.0

        # Flux continuity
        A1[-1, :] = 0
        A1[-1, -1] = 1
        A1[-1, -2] = -1
        b1[-1] = (D2/D1) * (u2_old[1] - u2_old[0])

        # Temperature continuity
        A2[0, :] = 0
        A2[0, 0] = 1
        b2[0] = u1_old[-1]

        u1_new = implicit_step(u1_old, A1, b1)
        u2_new = implicit_step(u2_old, A2, b2)

        if np.abs(u1_new[-1] - u2_new[0]) < tol:
            break

        u1_old = u1_new
        u2_old = u2_new

    u_new = np.concatenate((u1_new[:-1], u2_new))
    return u_new

# Pre-build matrices for left and right domains (size varies!)
N1 = mid + 1
N2 = Nx - mid

A1_template = build_matrix(N1, D1, dx, dt)
A2_template = build_matrix(N2, D2, dx, dt)

# Initialize solutions
u_explicit = u_init.copy()
u_implicit = u_init.copy()

# Storage for plotting every plot_freq steps
plot_freq = 200
times = []
u_explicit_snapshots = []
u_implicit_snapshots = []

for n in range(Nt):
    # Copy matrices to avoid overwriting rows changed for BCs
    A1 = A1_template.copy()
    A2 = A2_template.copy()
    
    u_explicit = explicit_coupling(u_explicit, A1, A2, dt, dx, mid)
    
    A1 = A1_template.copy()
    A2 = A2_template.copy()
    u_implicit = implicit_coupling(u_implicit, A1, A2, dt, dx, mid)
    
    if n % plot_freq == 0:
        times.append(n*dt)
        u_explicit_snapshots.append(u_explicit.copy())
        u_implicit_snapshots.append(u_implicit.copy())

# --- Interactive plot with slider ---
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(bottom=0.2)

line_exp, = ax.plot(x, u_explicit_snapshots[0], 'r-', label='Explicit')
line_imp, = ax.plot(x, u_implicit_snapshots[0], 'b--', label='Implicit')
ax.set_title("Temperature profile over domain at different times")
ax.set_xlabel("x")
ax.set_ylabel("Temperature")
ax.legend()

# Slider axis
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(ax_slider, 'Time', 0, len(times)-1, valinit=0, valstep=1)

def update(val):
    idx = int(slider.val)
    line_exp.set_ydata(u_explicit_snapshots[idx])
    line_imp.set_ydata(u_implicit_snapshots[idx])
    ax.set_title(f"Temperature profile at t={times[idx]:.3f}")
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.show()

# Plot difference at final time
plt.figure()
plt.plot(x, np.abs(u_explicit - u_implicit), 'k-', label='|Explicit - Implicit|')
plt.title("Difference between explicit and implicit solutions at final time")
plt.xlabel("x")
plt.ylabel("Absolute error")
plt.legend()
plt.show()
