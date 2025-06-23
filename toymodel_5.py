import numpy as np
import matplotlib.pyplot as plt

# === Generic System Parameters ===
# These constants define the behavior of our generic non-linear system.
# Adjust them to explore different dynamics.
A = 0.1
B = 1.0
C = 0.5   # The "non-differentiable" point for the absolute value term
D = 0.05
E = 0.2

# === Time Settings ===
DT = 0.1           # Time step for numerical integration
NUM_STEPS = 1000   # Total number of simulation steps

# === Coupling Frequency Setting ===
# CPL_FREQ determines how often the "lagged" values are updated in the staggered scheme.
# CPL_FREQ = 1 means coupling at every time step (effectively no additional lag beyond Euler's).
# CPL_FREQ > 1 introduces a true staggering/partitioning effect.
CPL_FREQ = 10 # Example: Update lagged values every 10 steps

# === Initial Conditions ===
u_initial = 1.0
v_initial = 2.0

# === Define the Generic Non-linear Functions ===
# These functions define the rates of change (du/dt and dv/dt).
# We use 'np.abs' to introduce a "sharply changing" behavior,
# and 'v**2' for another non-linearity.

def d_u_dt_func(u, v_coupling):
    """
    Rate of change for variable u.
    v_coupling: This will be either v_lagged (staggered) or v_current (synchronous).
    """
    return -A * u + B * np.abs(v_coupling - C)

def d_v_dt_func(v, u_coupling):
    """
    Rate of change for variable v.
    u_coupling: This will be either u_lagged (staggered) or u_current (synchronous).
    """
    return -D * v**2 + E * u_coupling

# === Data Storage ===
time_axis = np.linspace(0, NUM_STEPS * DT, NUM_STEPS + 1)

# History for Staggered (Partitioned) Coupling
u_staggered_history = []
v_staggered_history = []

# History for Synchronous Explicit Coupling
u_sync_history = []
v_sync_history = []

# === Simulation Loop: Staggered (Partitioned) Coupling ===
print(f"Running Staggered Simulation with CPL_FREQ = {CPL_FREQ}...")

u_staggered = u_initial
v_staggered = v_initial

# Initialize lagged values for the staggered scheme
u_lagged_for_v = u_initial
v_lagged_for_u = v_initial

u_staggered_history.append(u_staggered)
v_staggered_history.append(v_staggered)

for k in range(NUM_STEPS):
    # Update lagged values ONLY at the coupling frequency
    if k % CPL_FREQ == 0:
        u_lagged_for_v = u_staggered
        v_lagged_for_u = v_staggered

    # Calculate rates of change using the current state and lagged values for coupling terms
    du_dt_staggered = d_u_dt_func(u_staggered, v_lagged_for_u)
    dv_dt_staggered = d_v_dt_func(v_staggered, u_lagged_for_v)

    # Update u and v using Explicit Euler
    u_staggered = u_staggered + du_dt_staggered * DT
    v_staggered = v_staggered + dv_dt_staggered * DT

    u_staggered_history.append(u_staggered)
    v_staggered_history.append(v_staggered)

# === Simulation Loop: Synchronous Explicit Coupling ===
print("Running Synchronous Simulation (always coupled at every step)...")

u_sync = u_initial
v_sync = v_initial

u_sync_history.append(u_sync)
v_sync_history.append(v_sync)

for k in range(NUM_STEPS):
    # Store current values for synchronous update
    u_current = u_sync
    v_current = v_sync

    # Calculate rates of change using values from the *beginning of the current step*
    du_dt_sync = d_u_dt_func(u_current, v_current)
    dv_dt_sync = d_v_dt_func(v_current, u_current)

    # Update u and v using Explicit Euler
    u_sync = u_current + du_dt_sync * DT
    v_sync = v_current + dv_dt_sync * DT

    u_sync_history.append(u_sync)
    v_sync_history.append(v_sync)


# === Plot Results ===
plt.figure(figsize=(14, 8))

# Plot for variable u
plt.subplot(2, 1, 1)
plt.plot(time_axis, u_staggered_history, label=f'u (Staggered, CPL_FREQ={CPL_FREQ})', color='red', linestyle='-', linewidth=1.5)
plt.plot(time_axis, u_sync_history, label='u (Synchronous)', color='blue', linestyle='--', linewidth=1.5)
plt.title('Comparison of Staggered vs. Synchronous Explicit Coupling')
plt.ylabel('Variable u')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Plot for variable v
plt.subplot(2, 1, 2)
plt.plot(time_axis, v_staggered_history, label=f'v (Staggered, CPL_FREQ={CPL_FREQ})', color='darkgreen', linestyle='-', linewidth=1.5)
plt.plot(time_axis, v_sync_history, label='v (Synchronous)', color='purple', linestyle='--', linewidth=1.5)
plt.xlabel('Time')
plt.ylabel('Variable v')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("output.png")

print(f"\nSimulation complete. Plot saved as 'staggering_effect_non_linear_cpl_freq.png'.")
print(f"Observe the differences between the solid (Staggered, CPL_FREQ={CPL_FREQ}) and dashed (Synchronous) lines.")
print(f"A higher CPL_FREQ value will typically lead to larger differences, as the lagged information becomes 'staler'.")
