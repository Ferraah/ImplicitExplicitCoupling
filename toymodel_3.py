import numpy as np
import matplotlib.pyplot as plt

# === Constants and Coupling Coefficients ===
alpha_Q = 15.0     # Heat exchange coefficient
alpha_tau = 1e-3   # Wind stress coefficient
beta_a = 0.05      # Atmosphere sensitivity to SST
beta_o = 0.01      # Ocean response to heat flux
gamma = 0.1        # Atmosphere feedback from wind stress
delta = 0.002      # Sea surface height response to wind stress
cpl_freq = 20      # Coupling frequency (explicit lagged update)

# === Time Settings ===
num_steps = 100
dt = 1.0  # Time step (arbitrary units)

# === Initialize State Variables ===
T_a = np.zeros(num_steps)    # Atmospheric temperature [K]
u_a = np.zeros(num_steps)    # Wind speed [m/s]
T_o = np.zeros(num_steps)    # Ocean SST [K]
eta = np.zeros(num_steps)    # Sea surface height anomaly [m]
Q = np.zeros(num_steps)      # Net heat flux [W/m^2]
tau = np.zeros(num_steps)    # Wind stress [N/m^2]

# === Error Diagnostics ===
err_flux_residual = np.zeros(num_steps)   # Error from expected ocean update
err_flux_mismatch = np.zeros(num_steps)   # Mismatch between used and recomputed flux
err_rel_flux_residual = np.zeros(num_steps)  # Relative version of residual error

# === Initial Conditions ===
T_a[0] = 300.0   # Air temperature
T_o[0] = 298.0   # Ocean temperature
u_a[0] = 10.0    # Wind speed
tau[0] = alpha_tau * u_a[0]**2  # Initial wind stress

# Lagged variables for coupling
prev_T_a = T_a[0]
prev_T_o = T_o[0]

# === Time Integration Loop (Explicit Lagged Method) ===
for k in range(1, num_steps):
    # Update lagged values at coupling frequency
    if k % cpl_freq == 0:
        prev_T_a = T_a[k-1]
        prev_T_o = T_o[k-1]

    # --- Flux calculation using lagged values ---
    Q[k] = alpha_Q * (prev_T_a - T_o[k-1])
    tau[k] = alpha_tau * u_a[k-1]**2

    # --- Ocean update ---
    T_o[k] = T_o[k-1] + beta_o * Q[k]
    eta[k] = eta[k-1] + delta * tau[k]

    # --- Atmosphere update ---
    T_a[k] = T_a[k-1] + beta_a * (prev_T_o - T_a[k-1]) + gamma * tau[k-1]

    # --- Wind update (feedback based on lagged SST) ---
    u_a[k] = u_a[k-1] + 0.05 * (prev_T_o - T_a[k-1])

    # --- Error diagnostics ---
    expected_T_o = T_o[k-1] + beta_o * Q[k]
    err_flux_residual[k] = np.abs(T_o[k] - expected_T_o)

    Q_expected = alpha_Q * (T_a[k-1] - T_o[k-1])
    err_flux_mismatch[k] = np.abs(Q[k] - Q_expected)

    delta_T_o = T_o[k] - T_o[k-1]
    err_rel_flux_residual[k] = err_flux_residual[k] / np.abs(delta_T_o) if delta_T_o != 0 else 0.0

# === Plot Results ===
plt.figure(figsize=(12, 8))

# Subplot 1: Temperatures
plt.subplot(3, 2, 1)
plt.plot(T_a, label='Air Temp $T_a$')
plt.plot(T_o, label='SST $T_o$')
plt.title('Temperatures')
plt.legend()

# Subplot 2: Heat Flux
plt.subplot(3, 2, 2)
plt.plot(Q, label='Heat Flux $Q$')
plt.title('Air-Sea Heat Flux')
plt.legend()

# Subplot 3: Wind Stress
plt.subplot(3, 2, 3)
plt.plot(tau, label='Wind Stress $\\tau$')
plt.title('Momentum Flux')
plt.legend()

# Subplot 4: Flux Residual Error
plt.subplot(3, 2, 4)
plt.plot(err_flux_residual, label='Ocean Update Error')
plt.title('Error: $T_o - (T_o + \\beta_o Q)$')
plt.legend()

# Subplot 5: Flux Mismatch
plt.subplot(3, 2, 5)
plt.plot(err_flux_mismatch, label='Flux Mismatch')
plt.title('Error: $Q^{used} - Q^{actual}$')
plt.legend()

# Subplot 6: Relative Error
plt.subplot(3, 2, 6)
plt.plot(err_rel_flux_residual, label='Relative Flux Residual')
plt.title('Relative Error in Ocean Update')
plt.legend()

plt.tight_layout()
plt.savefig("output.png")

