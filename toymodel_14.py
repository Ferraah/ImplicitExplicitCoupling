import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
L = 1.0
Nx = 100
dx = L / (Nx - 1)
x = np.linspace(0, L, Nx)

D1, D2 = 0.1, 0.1
dt, T = 0.0002, 0.5
Nt = int(T / dt)
interface_idx = Nx // 2

# --- Initial condition ---
u0 = np.exp(-100 * (x - 0.5)**2)

def diffusion_step(u, D, dx, dt):
    """Vectorized diffusion step."""
    u_new = u.copy()
    coeff = D * dt / dx**2
    u_new[1:-1] += coeff[1:-1] * (u[2:] - 2 * u[1:-1] + u[:-2])
    return u_new

# --- Monolithic Reference ---
u_monolithic = np.zeros((Nt + 1, Nx))
u_monolithic[0] = u0.copy()
u = u0.copy()

for n in range(1, Nt + 1):
    D = np.where(x <= L / 2, D1, D2)
    u = diffusion_step(u, D, dx, dt)
    u[0], u[-1] = u[1], u[-2]  # Neumann (zero-gradient)
    u_monolithic[n] = u

# --- Explicit Coupling ---
u_explicit = np.zeros((Nt + 1, Nx))
uL = u0[:interface_idx + 1].copy()
uR = u0[interface_idx:].copy()
u_explicit[0] = np.concatenate((uL[:-1], uR))
exchange_freq = 1

for n in range(1, Nt + 1):
    # Diffuse
    uL[1:-1] += D1 * dt / dx**2 * (uL[2:] - 2 * uL[1:-1] + uL[:-2])
    uR[1:-1] += D2 * dt / dx**2 * (uR[2:] - 2 * uR[1:-1] + uR[:-2])

    # Exchange at interface
    if n % exchange_freq == 0:
        fluxL = D1 * (uL[-1] - uL[-2]) / dx
        fluxR = D2 * (uR[1] - uR[0]) / dx
        interface_flux = 0.5 * (fluxL + fluxR)
        uL[-1] = uL[-2] + dx * interface_flux / D1
        uR[0]  = uR[1] - dx * interface_flux / D2

    uL[0] = uL[1]
    uR[-1] = uR[-2]

    # Store combined
    u_combined = np.concatenate((uL[:-1], uR))
    u_combined[0], u_combined[-1] = u_combined[1], u_combined[-2]
    u_explicit[n] = u_combined

# --- Implicit Coupling with Schwarz iterations ---
u_implicit = np.zeros((Nt + 1, Nx))
uL = u0[:interface_idx + 1].copy()
uR = u0[interface_idx:].copy()
u_implicit[0] = np.concatenate((uL[:-1], uR))
max_iters = 1
tol = 1.e-8

for n in range(1, Nt + 1):
    # Save initial condition
    uL0 = uL.copy()
    uR0 = uR.copy()
    
    # Initial interface guess from previous time step
    interface_flux = D1 * (uL[-1] - uL[-2]) / dx

    for _ in range(max_iters):
        # Reset to initial state
        uL_new = uL0.copy()
        uR_new = uR0.copy()

        # Apply Neumann BCs at domain boundaries
        uL_new[0] = uL_new[1]
        uR_new[-1] = uR_new[-2]

        # Use current interface_flux to apply BC at interface
        uL_new[-1] = uL_new[-2] + dx * interface_flux / D1
        uR_new[0]  = uR_new[1] - dx * interface_flux / D2

        # Diffuse over one full dt
        uL_new[1:-1] += D1 * dt / dx**2 * (uL_new[2:] - 2 * uL_new[1:-1] + uL_new[:-2])
        uR_new[1:-1] += D2 * dt / dx**2 * (uR_new[2:] - 2 * uR_new[1:-1] + uR_new[:-2])

        # Recompute fluxes for next iteration
        fluxL_new = D1 * (uL_new[-1] - uL_new[-2]) / dx
        fluxR_new = D2 * (uR_new[1] - uR_new[0]) / dx
        new_interface_flux = 0.5 * (fluxL_new + fluxR_new)

        err =  abs(new_interface_flux - interface_flux)
        # Check convergence on interface flux
        if err < tol:
            print(_, err)
            break
        interface_flux = new_interface_flux

    uL = uL_new
    uR = uR_new
    u_combined = np.concatenate((uL[:-1], uR))
    u_combined[0], u_combined[-1] = u_combined[1], u_combined[-2]
    u_implicit[n] = u_combined


# --- Plot results ---
time_indices = [0, Nt//4, Nt//2, Nt-1]
labels = ['Monolithic', 'Explicit', 'Implicit']
styles = ['-', '--', ':']
colors = ['k', 'r', 'b']

fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
axs = axs.ravel()

for i, tidx in enumerate(time_indices):
    axs[i].plot(x, u_monolithic[tidx], styles[0], color=colors[0], label=labels[0])
    axs[i].plot(x, u_explicit[tidx], styles[1], color=colors[1], label=labels[1])
    axs[i].plot(x, u_implicit[tidx], styles[2], color=colors[2], label=labels[2])
    axs[i].set_title(f"t = {tidx * dt:.3f} s")
    axs[i].legend()

print(abs(np.sum(u_monolithic-u_explicit)/np.sum(u_monolithic)*100),"%")
print(abs(np.sum(u_monolithic-u_implicit)/np.sum(u_monolithic)*100),"%")
fig.suptitle("Coupled Diffusion Comparison (No Dirichlet BCs)")
plt.tight_layout()
plt.savefig("output.png")
