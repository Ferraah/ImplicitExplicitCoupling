import numpy as np
import matplotlib.pyplot as plt

def calculate_error(T_o, T_a, beta_o, Q):
    return abs(T_o - (T_a + beta_o*Q)) 

# Constants and coupling coefficients
alpha_Q = 15.0     # heat exchange coefficient
alpha_tau = 1e-3   # wind stress coefficient
beta_a = 0.05      # atmospheric sensitivity to SST
beta_o = 0.01      # ocean response to heat flux
gamma = 0.1        # atmosphere feedback from wind stress
delta = 0.002      # sea surface height response to wind stress
cpl_freq = 20

# Time stepping
num_steps = 100
dt = 1.0  # arbitrary units

# State variables
T_a = np.zeros(num_steps)    # Atmospheric temperature
u_a = np.zeros(num_steps)    # Wind speed
T_o = np.zeros(num_steps)    # Sea surface temperature
eta = np.zeros(num_steps)    # Sea surface height
Q = np.zeros(num_steps)      # Heat flux
tau = np.zeros(num_steps)    # Wind stress
err = np.zeros(num_steps)    # Wind stress

# Initial conditions
T_a[0] = 300.0  # Kelvin
T_o[0] = 298.0  # Kelvin
u_a[0] = 10.0   # m/s
err[0] = calculate_error(T_o[0], T_a[0], beta_o, Q[0])

prev_T_a = T_a[0]
prev_T_o = T_o[0]

for k in range(1, num_steps):
    if k%cpl_freq == 0:
        prev_T_a = T_a[k-1]
        prev_T_o = T_o[k-1]

    # Compute lagged fluxes (coupling delay by 1 step)
    Q[k] = alpha_Q * (prev_T_a - T_o[k-1])

    # Update ocean: SST and sea surface height
    T_o[k] = T_o[k-1] + beta_o * Q[k]
    eta[k] = eta[k-1] + delta * tau[k]

    tau[k] = alpha_tau * u_a[k-1]**2
    # Optional: simple wind feedback (e.g., strengthening due to SST gradient)
    u_a[k] = u_a[k-1] + 0.05 * (prev_T_o - T_a[k-1])
    # Update atmosphere: air temperature and wind speed
    T_a[k] = T_a[k-1] + beta_a * (prev_T_o - T_a[k-1]) + gamma * tau[k-1]


    # Check coherence
    err[k] = calculate_error(T_o[k], T_a[k], beta_o, Q[k])
    print(err[k])


# Plotting results
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.plot(T_a, label='Air Temp $T_a$')
plt.plot(T_o, label='SST $T_o$')
plt.legend()
plt.title('Temperatures')

plt.subplot(2, 2, 2)
plt.plot(Q, label='Heat Flux $Q$')
plt.title('Air-Sea Heat Flux')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(tau, label='Wind Stress $\\tau$')
plt.title('Momentum Flux')
plt.legend()

# plt.subplot(2, 2, 4)
# plt.plot(eta, label='Sea Surface Height $\\eta$')
# plt.title('SSH Anomaly')
# plt.legend()

plt.subplot(2, 2, 4)
plt.plot(err, label='error')
plt.title('Coherency')
plt.legend()

plt.tight_layout()
plt.savefig("output.png")
