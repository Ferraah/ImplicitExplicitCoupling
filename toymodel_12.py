import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 1.0        # domain half-length
Nx = 50        # points per domain
dx = L / Nx
xA = np.linspace(0, L, Nx+1)     # model A domain [0, L]
xB = np.linspace(L, 2*L, Nx+1)   # model B domain [L, 2L]

T = 0.5       # total time
dt = 0.0001    # timestep
Nt = int(T/dt)

# Model parameters
v = 2.0         # advection speed (Model A)
DA = 0.01       # diffusion coeff (Model A)
DB = 0.001      # diffusion coeff (Model B)
alpha = 0.1   # nonlinearity coeff (Model A)

# Coupling parameters
coupling_freq = 10   # explicit coupling frequency in timesteps
max_iter = 2        # max iterations for implicit scheme per coupling window
tol = 1e-5           # implicit convergence tolerance
implicit = True     # True for implicit, False for explicit

# Initialize fields
uA = np.zeros(Nx+1)  # Model A
uB = np.zeros(Nx+1)  # Model B

# Initial conditions: small perturbation near interface
uA[:] = 0.001*np.exp(-100*(xA - 0.8*L)**2)
uB[:] = 0.001*np.exp(-100*(xB - 1.2*L)**2)

def step_model_A(uA, uA_interface_val, dt):
    # Forward Euler advection-diffusion-reaction with Dirichlet BC at interface (x=L)
    u_new = uA.copy()
    for i in range(1, Nx):
        adv = -v * (uA[i] - uA[i-1]) / dx
        diff = DA * (uA[i-1] - 2*uA[i] + uA[i+1]) / dx**2
        react = -alpha * uA[i]**3
        u_new[i] = uA[i] + dt*(adv + diff + react)
    # Boundary conditions
    u_new[0] = 0.0    # fixed BC at x=0
    u_new[Nx] = uA_interface_val  # Dirichlet BC from coupling at x=L
    return u_new

def step_model_B(uB, flux_at_interface, dt):
    # Forward Euler diffusion with Neumann BC (flux) at interface x=L
    u_new = uB.copy()
    for i in range(1, Nx):
        diff = DB * (uB[i-1] - 2*uB[i] + uB[i+1]) / dx**2
        u_new[i] = uB[i] + dt*diff
    # Boundary conditions
    u_new[0] = uB[1] + dx * flux_at_interface / DB  # Neumann at interface x=L (left boundary of B)
    u_new[Nx] = 0.0  # fixed BC at x=2L
    return u_new

def compute_flux_A(uA):
    # Compute flux at interface x=L (right boundary of A domain)
    # Using backward difference for gradient at Nx
    dudx = (uA[Nx] - uA[Nx-1]) / dx
    flux = -DA * dudx
    return flux

def explicit_coupling(uA, uB):
    # In explicit scheme, exchange interface data every coupling_freq timesteps
    # Hold old data in between
    uA_interface_val = uB[0]  # Dirichlet BC for A from B
    flux_at_interface = compute_flux_A(uA)  # flux from A for B
    
    return uA_interface_val, flux_at_interface

def implicit_coupling(uA, uB, dt):
    # Iterative coupling to converge interface values within tolerance
    uA_iter = uA.copy()
    uB_iter = uB.copy()
    
    for it in range(max_iter):
        # Use last B interface value as BC for A
        uA_interface_val = uB_iter[0]
        uA_new = step_model_A(uA_iter, uA_interface_val, dt)
        
        # Compute flux from updated A
        flux_at_interface = compute_flux_A(uA_new)
        
        # Step B using this flux
        uB_new = step_model_B(uB_iter, flux_at_interface, dt)
        
        # Check convergence at interface (difference of Dirichlet BC for A and B[0])
        diff = np.abs(uB_new[0] - uA_new[Nx])
        uA_iter = uA_new
        uB_iter = uB_new
        
        if diff < tol:
            break
    return uA_iter, uB_iter, it+1, diff

# Storage for error at interface
interface_errors = []

# For explicit coupling, store last interface values
last_uA_interface = uB[0]
last_flux = compute_flux_A(uA)

# Storage for time evolution
uA_history = []
uB_history = []

for n in range(Nt):
    if implicit:
        uA, uB, nit, diff = implicit_coupling(uA, uB, dt)
        interface_errors.append(diff)
    else:
        if n % coupling_freq == 0:
            last_uA_interface, last_flux = explicit_coupling(uA, uB)
        uA = step_model_A(uA, last_uA_interface, dt)
        uB = step_model_B(uB, last_flux, dt)
        error = np.abs(uA[Nx] - uB[0])
        interface_errors.append(error)
    # Store current fields
    uA_history.append(uA.copy())
    uB_history.append(uB.copy())

uA_history = np.array(uA_history)  # shape: (Nt, Nx+1)
uB_history = np.array(uB_history)  # shape: (Nt, Nx+1)

# Plot space-time evolution for Model A
plt.figure(figsize=(8,4))
plt.imshow(uA_history.T, aspect='auto', origin='lower',
           extent=[0, T, xA[0], xA[-1]], cmap='viridis')
plt.colorbar(label='uA')
plt.xlabel('Time')
plt.ylabel('x (Model A)')
plt.title('Space-time evolution: Model A')

# Plot space-time evolution for Model B
plt.figure(figsize=(8,4))
plt.imshow(uB_history.T, aspect='auto', origin='lower',
           extent=[0, T, xB[0], xB[-1]], cmap='viridis')
plt.colorbar(label='uB')
plt.xlabel('Time')
plt.ylabel('x (Model B)')
plt.title('Space-time evolution: Model B')

# Plot final fields
plt.figure(figsize=(10,4))
plt.plot(xA, uA, label='Model A')
plt.plot(xB, uB, label='Model B')
plt.axvline(L, color='k', linestyle='--', label='Interface')
plt.title(f"Final fields, coupling: {'Implicit' if implicit else 'Explicit'}")
plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.grid()

plt.figure()
plt.plot(np.arange(Nt)*dt, interface_errors)
plt.yscale('log')
plt.title('Interface error over time')
plt.xlabel('Time')
plt.ylabel('|uA(L) - uB(L)|')
plt.grid()

plt.savefig("output.png")
