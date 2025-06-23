import numpy as np
import matplotlib.pyplot as plt

# === Simplified Constants ===
# Physical constants representing simplified physics
ALPHA_Q = 10.0      # Heat exchange coefficient [W/(m^2 K)]
SIGMA = 5.67e-8     # Stefan-Boltzmann constant [W/(m^2 K^4)]

# Forcing terms (e.g., solar radiation) - NOW MEAN VALUES
R_A_MEAN = 150.0    # Mean net radiation absorbed by atmosphere [W/m^2]
R_O_MEAN = 340.0    # Mean net radiation absorbed by ocean [W/m^2]

# Amplitudes for the seasonal cycle
# These values are illustrative and can be adjusted for desired variability
R_A_AMPLITUDE = 15.0 # Seasonal amplitude for atmosphere radiation [W/m^2]
R_O_AMPLITUDE = 70.0 # Seasonal amplitude for ocean radiation [W/m^2]

# Model-specific parameters
EPSILON_A = 0.5     # Atmosphere's effective emissivity
EPSILON_O = 0.9     # Ocean's effective emissivity
C_A = 1.0e6         # Heat capacity of the atmosphere per unit area [J/(m^2 K)]
C_O = 4.0e7         # Heat capacity of the ocean mixed layer per unit area [J/(m^2 K)]

# === Coupling and Time Settings ===
CPL_FREQ = 20

# Define total simulation time in years
NUM_YEARS = 10

# Time step is 1 day.
SECONDS_PER_STEP = 24 * 3600 # Time step in seconds (1 day)
DAYS_IN_YEAR = 365.0 # For calculating seasonal cycle
NUM_STEPS = int(NUM_YEARS * DAYS_IN_YEAR) # Total number of steps

# === Initial Conditions ===
T_a_initial = 250.0 # Initial atmospheric temperature [K]
T_o_initial = 280.0 # Initial sea surface temperature [K]

# === Data Storage ===
plot_indices = np.arange(0, NUM_STEPS + 1, 1)
T_a_history = []
T_o_history = []
time_axis = np.linspace(0, NUM_YEARS, len(plot_indices))

# === Main Simulation Loop ===

# Initialize the "true" model states
T_a = T_a_initial
T_o = T_o_initial

# Initialize the "lagged" values that each model component will see.
T_a_lagged_for_ocean = T_a_initial
T_o_lagged_for_atmos = T_o_initial

# Store initial values for plotting
T_a_history.append(T_a)
T_o_history.append(T_o)

for k in range(1, NUM_STEPS + 1):
    # Calculate current time in days for seasonal forcing
    current_day_of_year = (k % DAYS_IN_YEAR)

    # 1. Update time-dependent forcing terms (seasonal cycle)
    # Using sine function to simulate annual cycle.
    # The phase (e.g., -DAYS_IN_YEAR/4) can be adjusted to align peaks with seasons.
    # Here, a simple sine wave means peak radiation around day 91 (spring), then again around day 273 (autumn)
    # If we want peak radiation in summer (e.g., day 172 for NH summer solstice), we could use (current_day_of_year - 172)
    # For simplicity, starting sine wave at 0 at Jan 1.
    R_A = R_A_MEAN + R_A_AMPLITUDE * np.sin(2 * np.pi * current_day_of_year / DAYS_IN_YEAR)
    R_O = R_O_MEAN + R_O_AMPLITUDE * np.sin(2 * np.pi * current_day_of_year / DAYS_IN_YEAR)

    # 2. Exchange information (update lagged values) ONLY at the coupling frequency.
    if k % CPL_FREQ == 0:
        T_a_lagged_for_ocean = T_a
        T_o_lagged_for_atmos = T_o

    # 3. Calculate heat fluxes USING ONLY THE LAGGED VALUES.
    heat_flux_into_atmos = ALPHA_Q * (T_o_lagged_for_atmos - T_a)
    heat_flux_out_of_ocean = ALPHA_Q * (T_o - T_a_lagged_for_ocean)

    # 4. Calculate the rate of temperature change for each system independently.
    dT_a_dt = (R_A - EPSILON_A * SIGMA * T_a**4 + heat_flux_into_atmos) / C_A
    dT_o_dt = (R_O - EPSILON_O * SIGMA * T_o**4 - heat_flux_out_of_ocean) / C_O

    # 5. Update "true" temperatures using the stable time step.
    T_a = T_a + dT_a_dt * SECONDS_PER_STEP
    T_o = T_o + dT_o_dt * SECONDS_PER_STEP

    # 6. Store results for plotting
    T_a_history.append(T_a)
    T_o_history.append(T_o)


# === Plot Results ===
plt.figure(figsize=(10, 5))

# Plot the daily data
plt.plot(time_axis, T_a_history, label='Atmosphere Temp ($T_a$)', color='skyblue', linewidth=1.5)
plt.plot(time_axis, T_o_history, label='Ocean SST ($T_o$)', color='navy', linewidth=1.5)

plt.title('Partitioned Coupling with Seasonal Forcing')
plt.xlabel('Time [Years]')
plt.ylabel('Temperature [K]')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim(0, NUM_YEARS)


plt.tight_layout()
plt.savefig("partitioned_coupling_seasonal.png")