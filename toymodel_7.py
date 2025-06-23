import numpy as np
import matplotlib.pyplot as plt

# === Problem Parameters ===
# Length of each 1D domain
L_LEFT = 1.0  # Length of the left domain [m]
L_RIGHT = 1.0 # Length of the right domain [m]

# Number of grid points for each domain
N_LEFT = 50   # Number of spatial grid points in the left domain
N_RIGHT = 50  # Number of spatial grid points in the right domain

# Thermal diffusivity for each domain [m^2/s]
K_LEFT = 0.01
K_RIGHT = 0.005 # Different diffusivity to show distinct behavior

# Interface coupling constant (e.g., heat transfer coefficient at interface)
ALPHA_INTERFACE = 1 # [W/(m^2 K)] simplified

# === Time Settings ===
DT = 0.01          # Time step for numerical integration [s]
NUM_STEPS = 10000   # Total number of simulation steps

# === Coupling Frequency Setting for Staggered Scheme ===
# CPL_FREQ = 1 means coupling at every time step.
# CPL_FREQ > 1 introduces a true staggering/partitioning effect at the interface.
CPL_FREQ = 100 # Example: Couple at every time step at the interface

# === Initial Conditions ===
# Initial temperature profiles for each domain
# Left domain: linearly decreasing
T_left_initial = np.linspace(300.0, 290.0, N_LEFT) # [K]
# Right domain: linearly increasing
T_right_initial = np.linspace(290.0, 300.0, N_RIGHT) # [K]

# === Spatial Discretization ===
dx_left = L_LEFT / (N_LEFT - 1)
dx_right = L_RIGHT / (N_RIGHT - 1)

# Ensure stability condition for explicit diffusion
# dt <= dx^2 / (2 * K)
if DT > (dx_left**2 / (2 * K_LEFT)) or DT > (dx_right**2 / (2 * K_RIGHT)):
    print(f"WARNING: Time step DT={DT} might be too large for stability.")
    print(f"  Recommended max DT for left: {dx_left**2 / (2 * K_LEFT)}")
    print(f"  Recommended max DT for right: {dx_right**2 / (2 * K_RIGHT)}")

# Spatial axes for plotting
x_left = np.linspace(0, L_LEFT, N_LEFT)
x_right = np.linspace(L_LEFT, L_LEFT + L_RIGHT, N_RIGHT) # Right domain starts where left ends

# === Define Diffusion and Interface Coupling Functions ===
def compute_diffusion(T, K, dx):
    """Computes the rate of change due to diffusion in a 1D array."""
    dT_dt_diffusion = np.zeros_like(T)
    # Interior points: (T[i+1] - 2*T[i] + T[i-1]) / dx^2
    dT_dt_diffusion[1:-1] = K * (T[2:] - 2 * T[1:-1] + T[:-2]) / (dx**2)
    return dT_dt_diffusion

def compute_interface_flux(T_left_boundary, T_right_boundary, alpha_interface):
    """Computes the heat flux across the interface."""
    return alpha_interface * (T_right_boundary - T_left_boundary) # Flux from right to left

# === Data Storage ===
# We'll store snapshots at certain intervals for clarity, or just initial/final
# For now, let's just compare the final profiles for clear differences.
# Or, we can track the interface values over time if needed.
# For this example, let's track the full profile at a few time points.
snapshot_times = [0, NUM_STEPS * DT / 4, NUM_STEPS * DT / 2, NUM_STEPS * DT] # Snapshots at 0%, 25%, 50%, 100% of simulation
snapshot_indices = [int(t / DT) for t in snapshot_times]

T_left_staggered_snapshots = {}
T_right_staggered_snapshots = {}
T_left_sync_snapshots = {}
T_right_sync_snapshots = {}


# === Simulation Loop: Staggered (Partitioned) Coupling ===
print(f"Running Staggered 1D Simulation (CPL_FREQ = {CPL_FREQ})...")

T_left_staggered = np.copy(T_left_initial)
T_right_staggered = np.copy(T_right_initial)

# Initialize lagged interface temperatures
T_left_interface_lagged_for_right = T_left_staggered[-1]
T_right_interface_lagged_for_left = T_right_staggered[0]

# Store initial snapshot
if 0 in snapshot_indices:
    T_left_staggered_snapshots[0] = np.copy(T_left_staggered)
    T_right_staggered_snapshots[0] = np.copy(T_right_staggered)

for k in range(NUM_STEPS):
    # Update lagged interface values ONLY at the coupling frequency
    if k % CPL_FREQ == 0:
        T_left_interface_lagged_for_right = T_left_staggered[-1]
        T_right_interface_lagged_for_left = T_right_staggered[0]

    # Calculate diffusion rates for both domains
    dT_left_dt_diff = compute_diffusion(T_left_staggered, K_LEFT, dx_left)
    dT_right_dt_diff = compute_diffusion(T_right_staggered, K_RIGHT, dx_right)

    # Calculate interface flux terms using lagged values
    # Flux from left to right at interface: proportional to (T_left[-1] - T_right_lagged)
    # Flux from right to left at interface: proportional to (T_right[0] - T_left_lagged)
    # For heat flux across the interface, we need to apply it to the boundary points.

    # Rate of change at the rightmost point of the left domain due to coupling
    # This is an outgoing flux from left domain to right domain
    # Simplified: a term proportional to (T_left - T_right_lagged) affecting T_left[-1]
    # And (T_right - T_left_lagged) affecting T_right[0]
    flux_staggered = compute_interface_flux(T_left_staggered[-1], T_right_staggered[0], ALPHA_INTERFACE)

    # Apply flux to boundary points
    # Left domain's last point experiences flux *out*
    # Right domain's first point experiences flux *in*
    # These are simplified boundary conditions for coupling.
    # More rigorous BCs would involve ghost cells or flux discretization.
    # Here, we treat it as an external forcing at the boundary.
    dT_left_dt_staggered = np.copy(dT_left_dt_diff)
    dT_right_dt_staggered = np.copy(dT_right_dt_diff)

    # The coupling terms need to be added to the rates of change at the interface points.
    # For a simple explicit Euler scheme, let's treat the interface points as having an additional source/sink.
    # This assumes the interface flux acts over an area, and the heat capacity of the boundary cell.
    # Simplified application: distribute flux effect over boundary cells.
    # Flux out of left domain's last cell: -flux_staggered / (rho * Cp * dx_left)
    # Flux into right domain's first cell: +flux_staggered / (rho * Cp * dx_right)
    # For simplicity, we just use arbitrary scaling here to represent transfer.
    # More formally, this would be based on energy conservation for the boundary cells.

    # A more common way for coupled diffusion is that the flux depends on the values *at the interface*
    # and these are the ones being exchanged.
    # Let's adjust the update rule at the interface for simplicity of comparison.
    # The interface points are coupled directly.

    # Update T_left boundary (rightmost point)
    dT_left_dt_staggered[-1] = K_LEFT * (T_left_staggered[-2] - T_left_staggered[-1]) / (dx_left**2) \
                                + ALPHA_INTERFACE * (T_right_interface_lagged_for_left - T_left_staggered[-1]) / (dx_left * 1.0) # simplified scaling

    # Update T_right boundary (leftmost point)
    dT_right_dt_staggered[0] = K_RIGHT * (T_right_staggered[1] - T_right_staggered[0]) / (dx_right**2) \
                               + ALPHA_INTERFACE * (T_left_interface_lagged_for_right - T_right_staggered[0]) / (dx_right * 1.0) # simplified scaling


    # Update temperatures using Explicit Euler
    T_left_staggered = T_left_staggered + dT_left_dt_staggered * DT
    T_right_staggered = T_right_staggered + dT_right_dt_staggered * DT

    # Store snapshots
    if k + 1 in snapshot_indices:
        T_left_staggered_snapshots[k + 1] = np.copy(T_left_staggered)
        T_right_staggered_snapshots[k + 1] = np.copy(T_right_staggered)


# === Simulation Loop: Synchronous Explicit Coupling ===
print("Running Synchronous 1D Simulation (always coupled at every step)...")

T_left_sync = np.copy(T_left_initial)
T_right_sync = np.copy(T_right_initial)

# Store initial snapshot
if 0 in snapshot_indices:
    T_left_sync_snapshots[0] = np.copy(T_left_sync)
    T_right_sync_snapshots[0] = np.copy(T_right_sync)

for k in range(NUM_STEPS):
    # Store current values for synchronous update
    T_left_current = np.copy(T_left_sync)
    T_right_current = np.copy(T_right_sync)

    # Calculate diffusion rates for both domains
    dT_left_dt_diff = compute_diffusion(T_left_current, K_LEFT, dx_left)
    dT_right_dt_diff = compute_diffusion(T_right_current, K_RIGHT, dx_right)

    dT_left_dt_sync = np.copy(dT_left_dt_diff)
    dT_right_dt_sync = np.copy(dT_right_dt_diff)

    # Apply synchronous coupling at the interface
    # This involves updating the boundary points using each other's *current* values
    dT_left_dt_sync[-1] = K_LEFT * (T_left_current[-2] - T_left_current[-1]) / (dx_left**2) \
                          + ALPHA_INTERFACE * (T_right_current[0] - T_left_current[-1]) / (dx_left * 1.0)

    dT_right_dt_sync[0] = K_RIGHT * (T_right_current[1] - T_right_current[0]) / (dx_right**2) \
                          + ALPHA_INTERFACE * (T_left_current[-1] - T_right_current[0]) / (dx_right * 1.0)


    # Update temperatures using Explicit Euler
    T_left_sync = T_left_sync + dT_left_dt_sync * DT
    T_right_sync = T_right_sync + dT_right_dt_sync * DT

    # Store snapshots
    if k + 1 in snapshot_indices:
        T_left_sync_snapshots[k + 1] = np.copy(T_left_sync)
        T_right_sync_snapshots[k + 1] = np.copy(T_right_sync)


# === Plot Results ===
plt.figure(figsize=(15, 10))

# Iterate through snapshots
for i, time_idx in enumerate(snapshot_indices):
    current_time = snapshot_times[i]
    if current_time == 0:
        title_suffix = f" (Initial State)"
    else:
        title_suffix = f" (Time = {current_time:.2f} s)"

    plt.subplot(len(snapshot_indices), 1, i + 1)
    plt.plot(x_left, T_left_staggered_snapshots[time_idx], color='red', linestyle='-', label=f'Left (Staggered, CPL_FREQ={CPL_FREQ})')
    plt.plot(x_right, T_right_staggered_snapshots[time_idx], color='red', linestyle='-', label='') # No label for right to avoid clutter
    plt.plot(x_left, T_left_sync_snapshots[time_idx], color='blue', linestyle='--', label='Left (Synchronous)')
    plt.plot(x_right, T_right_sync_snapshots[time_idx], color='blue', linestyle='--', label='') # No label for right to avoid clutter

    # Draw a vertical line at the interface
    plt.axvline(x=L_LEFT, color='gray', linestyle=':', label='Interface')

    plt.title(f'Temperature Profile{title_suffix}')
    plt.xlabel('Position [m]')
    plt.ylabel('Temperature [K]')
    plt.legend(loc='lower right', fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.suptitle('Staggering Influence on 1D Coupled Domains (Heat Diffusion)', y=1.02, fontsize=16)
plt.savefig("output.png")

print(f"\nSimulation complete. Plot saved as 'staggering_1d_interface_analysis.png'.")
print(f"This plot shows the temperature profiles in two 1D coupled domains at different time snapshots.")
print(f"The solid lines represent the staggered coupling, and the dashed lines represent the synchronous coupling.")
print(f"Observe the differences, especially at and around the interface (gray dotted line), as time progresses.")
print(f"Try changing CPL_FREQ to values greater than 1 (e.g., 5, 10) to see a more pronounced staggering effect, where the solutions will diverge more significantly, particularly near the coupling interface.")
