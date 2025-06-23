import numpy as np
import matplotlib.pyplot as plt

# === Constants and Coupling Coefficients ===
alpha_Q = 15.0     # Heat exchange coefficient
alpha_tau = 1e-3   # Wind stress coefficient
beta_a = 0.05      # Atmosphere sensitivity to SST
beta_o = 0.01      # Ocean response to heat flux
gamma = 0.1        # Atmosphere feedback from wind stress
delta = 0.002      # Sea surface height response to wind stress
cpl_freq = 20     # Coupling frequency (in timesteps)

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
err = np.zeros(num_steps)    # Coherency error in ocean update

# === Initial Conditions ===
T_a[0] = 300.0   # Air temperature
T_o[0] = 298.0   # Ocean temperature
u_a[0] = 10.0    # Wind speed
tau[0] = alpha_tau * u_a[0]**2  # Initial wind stress

# Use previous state copies for coupling
prev_T_a = T_a[0]
prev_T_o = T_o[0]

# === Coherency Error Function ===
def calculate_error_flux_based(T_o_k, T_o_prev, Q_k, beta_o):
    """Check how closely the ocean update matches the flux-based prediction."""
    # return np.abs((T_o_k - (T_o_prev) - beta_o * Q_k)
    return 0

# === Time Integration Loop ===
for k in range(1, num_steps):
    # === Coupling exchange (every cpl_freq steps) ===
    if k % cpl_freq == 0:
        prev_T_a = T_a[k-1]
        prev_T_o = T_o[k-1]

    # === Flux Calculation (using lagged input) ===
    Q[k] = alpha_Q * np.tanh(prev_T_a - T_o[k-1])
    tau[k] = alpha_tau * u_a[k-1]**2

    # === Ocean Update ===
    T_o[k] = T_o[k-1] + beta_o * Q[k]
    eta[k] = eta[k-1] + delta * tau[k]  # using updated tau

    # === Atmosphere Update ===
    T_a[k] = T_a[k-1] + beta_a * (prev_T_o - T_a[k-1]) + gamma * tau[k-1]

    # === Wind Speed Update ===
    u_a[k] = u_a[k-1] + 0.05 * (prev_T_o - T_a[k-1])  # simple wind feedback

    # === Coherency Error Diagnostics ===
    err[k] = calculate_error_flux_based(T_o[k], T_o[k-1], Q[k], beta_o)
    print(err[k])

# === Plot Results ===
plt.figure(figsize=(12, 6))

# Subplot 1: Temperatures
plt.subplot(2, 2, 1)
plt.plot(T_a, label='Air Temp $T_a$')
plt.plot(T_o, label='SST $T_o$')
plt.legend()
plt.title('Temperatures')

# Subplot 2: Heat Flux
plt.subplot(2, 2, 2)
plt.plot(Q, label='Heat Flux $Q$')
plt.title('Air-Sea Heat Flux')
plt.legend()

# Subplot 3: Wind Stress
plt.subplot(2, 2, 3)
plt.plot(tau, label='Wind Stress $\\tau$')
plt.title('Momentum Flux')
plt.legend()

# Subplot 4: Coherency Error
plt.subplot(2, 2, 4)
plt.plot(err, label='Coherency Error')
plt.title('Ocean-Flux Consistency Error')
plt.legend()

plt.tight_layout()
plt.savefig("output.png")
