import numpy as np
import matplotlib.pyplot as plt

# === Time Settings ===
NUM_YEARS = 1    # Total simulation duration in years
DAYS_IN_YEAR = 365.0 # For calculating seasonal cycle
DT = 24 * 3600     # Time step in seconds (1 day) - represents one model step
NUM_STEPS = int(NUM_YEARS * DAYS_IN_YEAR) # Total number of simulation steps

# === Coupled Atmosphere-Ocean Box Model Constants ===
# Physical constants representing simplified physics
ALPHA_Q = 10.0      # Heat exchange coefficient [W/(m^2 K)]
SIGMA = 5.67e-8     # Stefan-Boltzmann constant [W/(m^2 K^4)]

# Forcing terms (e.g., solar radiation) - MEAN VALUES
R_A_MEAN = 150.0    # Mean net radiation absorbed by atmosphere [W/m^2]
R_O_MEAN = 340.0    # Mean net radiation absorbed by ocean [W/m^2]

# Amplitudes for the seasonal cycle of radiation
R_A_AMPLITUDE = 15.0 # Seasonal amplitude for atmosphere radiation [W/m^2]
R_O_AMPLITUDE = 70.0 # Seasonal amplitude for ocean radiation [W/m^2]

# Model-specific parameters
EPSILON_A = 0.5     # Atmosphere's effective emissivity
EPSILON_O = 0.9     # Ocean's effective emissivity
C_A = 1.0e6         # Heat capacity of the atmosphere per unit area [J/(m^2 K)]
C_O = 4.0e7         # Heat capacity of the ocean mixed layer per unit area [J/(m^2 K)]

# === Carbon Cycle Parameters ===
CO2_INITIAL = 280.0             # Initial atmospheric CO2 concentration [ppm]
CO2_REFERENCE_FORCING = 280.0   # Reference CO2 for radiative forcing [ppm]
# Radiative forcing factor (e.g., IPCC value for CO2, approximately 5.35 W/m^2 per doubling)
# RF = alpha * ln(C/C0) => alpha = 5.35 / ln(2)
CO2_RADIATIVE_FORCING_FACTOR = 5.35 / np.log(2) # W/m^2
CO2_OCEAN_EXCHANGE_RATE = 0.05 / (DAYS_IN_YEAR * 3600 * 24) # ppm per K per second (simplified flux)
T_OCEAN_CO2_EQ_REF = 285.0      # Reference ocean temperature for CO2 equilibrium [K]

# === Ice Albedo Feedback Parameters ===
ICE_INITIAL = 0.5               # Initial ice extent (dimensionless, 0 to 1)
ICE_MELT_TEMP_THRESHOLD = 273.15 + 2 # Atmospheric temp above which ice melts rapidly [K] (e.g. 2C above freezing)
ICE_FREEZE_TEMP_THRESHOLD = 273.15 - 2 # Atmospheric temp below which ice freezes rapidly [K] (e.g. 2C below freezing)
ICE_CHANGE_RATE = 0.0001 / (DAYS_IN_YEAR * 3600 * 24) # Rate of ice extent change per second (dimensionless)
ALBEDO_ICE_EFFECT = 0.2         # Maximum reduction in absorbed ocean radiation due to ice (0 to 1)

# === Stochastic Variability Parameters ===
# Standard deviation of the random noise added to radiation forcing
STOCHASTIC_NOISE_STD_DEV = 5.0 # W/m^2 - Adjust for more or less randomness
np.random.seed(42) # For reproducibility of random numbers


# === Coupling Frequency Setting ===
# CPL_FREQ determines how often the "lagged" values are updated in the staggered scheme.
# CPL_FREQ = 1 means coupling at every time step (effectively daily exchange).
# CPL_FREQ > 1 introduces a true staggering/partitioning effect (e.g., weekly, monthly coupling).
CPL_FREQ = 10 # Example: Couple at every daily time step

# === Initial Conditions ===
T_a_initial = 250.0 # Initial atmospheric temperature [K]
T_o_initial = 280.0 # Initial sea surface temperature [K]
CO2_a_initial = CO2_INITIAL # Initial atmospheric CO2 [ppm]
Ice_extent_initial = ICE_INITIAL # Initial ice extent

# === Define the Coupled Atmosphere-Ocean Box Model Functions ===
def dT_a_dt_func(T_a, T_o_coupling, CO2_a_coupling, R_A_current):
    """
    Rate of change for Atmospheric Temperature (Ta).
    T_o_coupling: Either T_o_lagged (staggered) or T_o_current (synchronous).
    CO2_a_coupling: Either CO2_a_lagged (staggered) or CO2_a_current (synchronous).
    R_A_current: Current absorbed radiation by atmosphere (includes seasonal and stochastic).
    """
    # Radiative forcing from CO2
    RF_CO2 = CO2_RADIATIVE_FORCING_FACTOR * np.log(CO2_a_coupling / CO2_REFERENCE_FORCING)

    heat_flux_into_atmos = ALPHA_Q * (T_o_coupling - T_a)
    return (R_A_current + RF_CO2 - EPSILON_A * SIGMA * T_a**4 + heat_flux_into_atmos) / C_A

def dT_o_dt_func(T_o, T_a_coupling, Ice_extent_coupling, R_O_current):
    """
    Rate of change for Ocean Temperature (To).
    T_a_coupling: Either T_a_lagged (staggered) or T_a_current (synchronous).
    Ice_extent_coupling: Either Ice_extent_lagged (staggered) or Ice_extent_current (synchronous).
    R_O_current: Current absorbed radiation by ocean (includes seasonal and stochastic).
    """
    # Effect of ice albedo on absorbed ocean radiation
    R_O_effective = R_O_current * (1 - ALBEDO_ICE_EFFECT * Ice_extent_coupling)

    heat_flux_out_of_ocean = ALPHA_Q * (T_o - T_a_coupling)
    return (R_O_effective - EPSILON_O * SIGMA * T_o**4 - heat_flux_out_of_ocean) / C_O

def dCO2_a_dt_func(CO2_a, T_o_coupling):
    """
    Rate of change for atmospheric CO2 concentration (CO2_a).
    Simplified model: Ocean releases CO2 when warmer than reference, absorbs when colder.
    """
    return CO2_OCEAN_EXCHANGE_RATE * (T_o_coupling - T_OCEAN_CO2_EQ_REF)

def dIce_extent_dt_func(Ice_extent, T_a_coupling):
    """
    Rate of change for ice extent.
    Simplified model: Ice grows below freezing threshold, melts above melting threshold.
    """
    rate = 0.0
    if T_a_coupling < ICE_FREEZE_TEMP_THRESHOLD:
        rate = ICE_CHANGE_RATE # Positive change for growth
    elif T_a_coupling > ICE_MELT_TEMP_THRESHOLD:
        rate = -ICE_CHANGE_RATE # Negative change for melt

    # Apply bounds to ice extent (0 to 1)
    if Ice_extent + rate * DT > 1.0:
        return (1.0 - Ice_extent) / DT # Prevents overshooting 1
    elif Ice_extent + rate * DT < 0.0:
        return (0.0 - Ice_extent) / DT # Prevents undershooting 0
    else:
        return rate


# === Data Storage ===
time_axis = np.linspace(0, NUM_YEARS, NUM_STEPS + 1) # Time axis in years

# History for Staggered (Partitioned) Coupling
T_a_staggered_history = []
T_o_staggered_history = []
CO2_a_staggered_history = []
Ice_extent_staggered_history = []

# History for Synchronous Explicit Coupling
T_a_sync_history = []
T_o_sync_history = []
CO2_a_sync_history = []
Ice_extent_sync_history = []

# === Simulation Loop: Staggered (Partitioned) Coupling ===
print(f"Running Staggered Simulation (CPL_FREQ = {CPL_FREQ})...")

T_a_staggered = T_a_initial
T_o_staggered = T_o_initial
CO2_a_staggered = CO2_a_initial
Ice_extent_staggered = Ice_extent_initial

# Initialize lagged values for the staggered scheme
T_a_lagged_for_ocean = T_a_initial
T_o_lagged_for_atmos = T_o_initial
CO2_a_lagged_for_forcing = CO2_a_initial
Ice_extent_lagged_for_albedo = Ice_extent_initial

T_a_staggered_history.append(T_a_staggered)
T_o_staggered_history.append(T_o_staggered)
CO2_a_staggered_history.append(CO2_a_staggered)
Ice_extent_staggered_history.append(Ice_extent_staggered)


for k in range(NUM_STEPS):
    # Calculate current day of year for seasonal forcing
    current_day_of_year = (k % DAYS_IN_YEAR)

    # Calculate time-dependent forcing terms (seasonal cycle + stochastic noise)
    # Generate common noise for both simulations at each step for fair comparison
    stochastic_noise = np.random.normal(0, STOCHASTIC_NOISE_STD_DEV)

    R_A_current_step = R_A_MEAN + R_A_AMPLITUDE * np.sin(2 * np.pi * current_day_of_year / DAYS_IN_YEAR) + stochastic_noise
    R_O_current_step = R_O_MEAN + R_O_AMPLITUDE * np.sin(2 * np.pi * current_day_of_year / DAYS_IN_YEAR) + stochastic_noise

    # Update lagged values ONLY at the coupling frequency
    if k % CPL_FREQ == 0:
        T_a_lagged_for_ocean = T_a_staggered
        T_o_lagged_for_atmos = T_o_staggered
        CO2_a_lagged_for_forcing = CO2_a_staggered
        Ice_extent_lagged_for_albedo = Ice_extent_staggered

    # Calculate rates of change using the current state and lagged values for coupling terms
    dT_a_dt_staggered = dT_a_dt_func(T_a_staggered, T_o_lagged_for_atmos, CO2_a_lagged_for_forcing, R_A_current_step)
    dT_o_dt_staggered = dT_o_dt_func(T_o_staggered, T_a_lagged_for_ocean, Ice_extent_lagged_for_albedo, R_O_current_step)
    dCO2_a_dt_staggered = dCO2_a_dt_func(CO2_a_staggered, T_o_lagged_for_atmos)
    dIce_extent_dt_staggered = dIce_extent_dt_func(Ice_extent_staggered, T_a_lagged_for_ocean) # Ice dependent on atmosphere

    # Update temperatures, CO2, and ice extent using Explicit Euler
    T_a_staggered = T_a_staggered + dT_a_dt_staggered * DT
    T_o_staggered = T_o_staggered + dT_o_dt_staggered * DT
    CO2_a_staggered = CO2_a_staggered + dCO2_a_dt_staggered * DT
    Ice_extent_staggered = Ice_extent_staggered + dIce_extent_dt_staggered * DT

    # Ensure Ice_extent stays within [0, 1] bounds
    Ice_extent_staggered = np.clip(Ice_extent_staggered, 0.0, 1.0)
    # Ensure CO2_a stays positive
    CO2_a_staggered = max(CO2_a_staggered, 0.0)


    T_a_staggered_history.append(T_a_staggered)
    T_o_staggered_history.append(T_o_staggered)
    CO2_a_staggered_history.append(CO2_a_staggered)
    Ice_extent_staggered_history.append(Ice_extent_staggered)

# === Simulation Loop: Synchronous Explicit Coupling ===
print("Running Synchronous Simulation (always coupled at every step)...")

T_a_sync = T_a_initial
T_o_sync = T_o_initial
CO2_a_sync = CO2_a_initial
Ice_extent_sync = Ice_extent_initial

T_a_sync_history.append(T_a_sync)
T_o_sync_history.append(T_o_sync)
CO2_a_sync_history.append(CO2_a_sync)
Ice_extent_sync_history.append(Ice_extent_sync)

# Re-seed for synchronous simulation to get the exact same random sequence
np.random.seed(42)

for k in range(NUM_STEPS):
    # Calculate current day of year for seasonal forcing
    current_day_of_year = (k % DAYS_IN_YEAR)

    # Calculate time-dependent forcing terms (seasonal cycle + stochastic noise)
    # Generate common noise for both simulations at each step for fair comparison
    stochastic_noise = np.random.normal(0, STOCHASTIC_NOISE_STD_DEV)

    R_A_current_step = R_A_MEAN + R_A_AMPLITUDE * np.sin(2 * np.pi * current_day_of_year / DAYS_IN_YEAR) + stochastic_noise
    R_O_current_step = R_O_MEAN + R_O_AMPLITUDE * np.sin(2 * np.pi * current_day_of_year / DAYS_IN_YEAR) + stochastic_noise

    # Store current values for synchronous update
    T_a_current = T_a_sync
    T_o_current = T_o_sync
    CO2_a_current = CO2_a_sync
    Ice_extent_current = Ice_extent_sync

    # Calculate rates of change using values from the *beginning of the current step*
    dT_a_dt_sync = dT_a_dt_func(T_a_current, T_o_current, CO2_a_current, R_A_current_step)
    dT_o_dt_sync = dT_o_dt_func(T_o_current, T_a_current, Ice_extent_current, R_O_current_step)
    dCO2_a_dt_sync = dCO2_a_dt_func(CO2_a_current, T_o_current)
    dIce_extent_dt_sync = dIce_extent_dt_func(Ice_extent_current, T_a_current)

    # Update temperatures, CO2, and ice extent using Explicit Euler
    T_a_sync = T_a_sync + dT_a_dt_sync * DT
    T_o_sync = T_o_sync + dT_o_dt_sync * DT
    CO2_a_sync = CO2_a_sync + dCO2_a_dt_sync * DT
    Ice_extent_sync = Ice_extent_sync + dIce_extent_dt_sync * DT

    # Ensure Ice_extent stays within [0, 1] bounds
    Ice_extent_sync = np.clip(Ice_extent_sync, 0.0, 1.0)
    # Ensure CO2_a stays positive
    CO2_a_sync = max(CO2_a_sync, 0.0)


    T_a_sync_history.append(T_a_sync)
    T_o_sync_history.append(T_o_sync)
    CO2_a_sync_history.append(CO2_a_sync)
    Ice_extent_sync_history.append(Ice_extent_sync)


# === Plot Results ===
plt.figure(figsize=(16, 12)) # Larger figure for 4 subplots

# Plot for Atmosphere Temperature
plt.subplot(2, 2, 1)
plt.plot(time_axis, T_a_staggered_history, label=f'Atmosphere (Staggered, CPL_FREQ={CPL_FREQ})', color='red', linestyle='-', linewidth=1.5)
plt.plot(time_axis, T_a_sync_history, label='Atmosphere (Synchronous)', color='blue', linestyle='--', linewidth=1.5)
plt.title('Atmosphere Temperature ($T_a$) with Stochastic Variability')
plt.ylabel('Temperature [K]')
plt.legend(fontsize=8)
plt.grid(True, linestyle='--', alpha=0.7)

# Plot for Ocean Temperature
plt.subplot(2, 2, 2)
plt.plot(time_axis, T_o_staggered_history, label=f'Ocean (Staggered, CPL_FREQ={CPL_FREQ})', color='darkgreen', linestyle='-', linewidth=1.5)
plt.plot(time_axis, T_o_sync_history, label='Ocean (Synchronous)', color='purple', linestyle='--', linewidth=1.5)
plt.title('Ocean Surface Temperature ($T_o$) with Stochastic Variability')
plt.ylabel('Temperature [K]')
plt.legend(fontsize=8)
plt.grid(True, linestyle='--', alpha=0.7)

# Plot for Atmospheric CO2
plt.subplot(2, 2, 3)
plt.plot(time_axis, CO2_a_staggered_history, label=f'Atmospheric CO2 (Staggered, CPL_FREQ={CPL_FREQ})', color='orange', linestyle='-', linewidth=1.5)
plt.plot(time_axis, CO2_a_sync_history, label='Atmospheric CO2 (Synchronous)', color='brown', linestyle='--', linewidth=1.5)
plt.title('Atmospheric CO2 Concentration with Stochastic Variability')
plt.xlabel('Time [Years]')
plt.ylabel('CO2 [ppm]')
plt.legend(fontsize=8)
plt.grid(True, linestyle='--', alpha=0.7)

# Plot for Ice Extent
plt.subplot(2, 2, 4)
plt.plot(time_axis, Ice_extent_staggered_history, label=f'Ice Extent (Staggered, CPL_FREQ={CPL_FREQ})', color='cyan', linestyle='-', linewidth=1.5)
plt.plot(time_axis, Ice_extent_sync_history, label='Ice Extent (Synchronous)', color='darkblue', linestyle='--', linewidth=1.5)
plt.title('Ice Extent with Stochastic Variability')
plt.xlabel('Time [Years]')
plt.ylabel('Ice Extent [0-1]')
plt.legend(fontsize=8)
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("output.png")

print(f"\nSimulation complete. Plot saved as 'coupled_box_model_full_feedback_staggering_effect_stochastic.png'.")
print(f"This plot compares the staggered and synchronous explicit coupling methods for a simplified atmosphere-ocean box model, now including a carbon cycle, ice albedo feedback, and stochastic variability.")
print(f"The stochastic noise, along with seasonal forcing, introduces more dynamic (less smooth) behavior. Observe the differences between the solid (Staggered, CPL_FREQ={CPL_FREQ}) and dashed (Synchronous) lines for all four variables.")
print(f"A higher CPL_FREQ value means less frequent data exchange between components, which can lead to larger deviations from the synchronous solution, potentially affecting accuracy or stability, especially with strong non-linearities or rapid changes in any of the coupled variables.")
