import numpy as np
import matplotlib.pyplot as plt
# Removed unused import

# Physical and numerical parameters
L_A, L_B = 1.0, 1.0
Nx_A, Nx_B = 50, 50
# alpha_A, alpha_B = 1.16e-4, 2.2e-5
alpha_A, alpha_B = 1.16e-4, 1.16e-4  # Same thermal diffusivity for both models
beta = 0 # nonlinearity factor
dx = L_A / (Nx_A - 1)
dt = 0.4 * dx**2 / max(alpha_A, alpha_B)

# Snapshots
num_snapshots = 10



def update_heat_explicit(T, alpha, dx, dt, n):
    T_new = T.copy()
    for i in range(1, len(T) - 1):
        T_new[i] = T[i] + alpha * dt / dx**2 * (T[i-1] - 2*T[i] + T[i+1])
    T_new[0] = T_new[1]       # Neumann BC left
    T_new[-1] = T_new[-2]     # Neumann BC right
    
    # Add nonlinear source term
    T_new += beta * dt * np.exp(-n*dt)
    return T_new

# Solve heat equation on the whole domain directly 
def monolithic_coupling(Nt):
    # Initial conditions: piecewise linear
    T_A = np.linspace(20.0, 20.0, Nx_A)
    T_B = np.linspace(10.0, 10.0, Nx_B)
    
    T_full = np.hstack((T_A, T_B))
    # Storage
    snapshots_history = []
    full_history = []      

    n_iter = 0
    # Time stepping loop
    for n in range(0, Nt):

        full_history.append(T_full)

        # Save snapshot
        if n % (Nt // (num_snapshots - 1)) == 0:
            snapshots_history.append(T_full.copy())
        
        T_full = update_heat_explicit(T_full, alpha_A, dx, dt, n)
        n_iter += 1

    print("Timesteps number: ", Nt)
    print("Total model runs: ", n_iter)

    # Plot all snapshots
    plt.figure(figsize=(10, 6))
    for i, snapshot in enumerate(snapshots_history):
        plt.plot(snapshot, label=f"Snapshot {i}")
    plt.xlabel("Spatial Index")
    plt.ylabel("Temperature")
    plt.title("Temperature Evolution Over Time")
    plt.legend()
    plt.grid()
    plt.savefig("output_monolithic.png")

    return full_history 

             
def implicit_coupling(Nt, cpl_frequency, max_iter, tol):
    # Fixed-point iteration parameters
    n_solves = 0

    snapshot_interval = Nt // (num_snapshots - 1)

    # Initial conditions: piecewise linear
    T_A = np.linspace(20.0, 20.0, Nx_A)
    T_B = np.linspace(10.0, 10.0, Nx_B)

    snapshots_history = []
    full_history = []      

    # Time stepping loop
    for n in range(0, Nt):

        full_history.append(np.hstack((T_A, T_B)))        

        if n % cpl_frequency == 0:
            T_A_lag = T_A.copy()
            T_B_lag = T_B.copy()
        # Else, use the ones retrieved at a previous coupling time

        # Save snapshot
        if n % snapshot_interval == 0:
            snapshots_history.append(np.concatenate((T_A, T_B)))

        T_if = 0.5 * (T_A_lag[-1] + T_B_lag[0])

        T_A_old = T_A.copy()
        T_B_old = T_B.copy()

        T_A_old[-1] = T_if 
        T_B_old[0] = T_if

        for iter_count in range(1, max_iter + 1):

            # Both models calculate in the same way temperature at the interface
            # using the avalable data from the other side

            # Calculate new state  after interface update
            T_A_check = update_heat_explicit(T_A_old, alpha_A, dx, dt, n)
            T_B_check = update_heat_explicit(T_B_old, alpha_B, dx, dt, n)
            n_solves += 1

            # Check dicrepancy at interface 
            err_a = np.sum(np.abs(T_A_check - T_A_old))
            err_b = np.sum(np.abs(T_B_check - T_B_old))

            if err_a < tol and err_b < tol:
                print(f"Converged after {iter_count} iterations at timestep {n}")
                break 
        
            T_A_old = T_A_check.copy()
            T_B_old = T_B_check.copy()

            T_if = 0.5 * (T_A_check[-1] + T_B_check[0])
            
            T_A_old[-1] = T_if 
            T_B_old[0] = T_if



        T_A = T_A_check.copy()
        T_B = T_B_check.copy()

    snapshots_history.append(np.concatenate((T_A, T_B)))
    print("Timesteps number: ", Nt)
    print("Total model runs: ", n_solves)
    # Plot all snapshots
    plt.figure(figsize=(10, 6))
    for i, snapshot in enumerate(snapshots_history):
        plt.plot(snapshot, label=f"Snapshot {i}")
    plt.xlabel("Spatial Index")
    plt.ylabel("Temperature")
    plt.title("Temperature Evolution Over Time")
    plt.legend()
    plt.grid()
    plt.savefig("output.png")

    return full_history

simulation_time = 1000
Nt1 = int(simulation_time//dt)
Nt2 = 10*Nt1


print("Running baseline")
baseline_impl = np.asarray(monolithic_coupling(Nt2), dtype=object)
baseline_expl = np.asarray(monolithic_coupling(Nt2), dtype=object)

print("\nRunning explicit")
hist_expl = np.asarray(implicit_coupling(Nt2, 1, 1, 1e-100), dtype=object)
print("\nRunning implicit")
hist_impl = np.asarray(implicit_coupling(Nt1, 1, 10, 1e-100), dtype=object)

# Retrievve one every 10 elements of baseline_impl
baseline_impl = baseline_impl[::10]

err_impl_time = np.abs(hist_impl-baseline_impl)/(Nx_A + Nx_B)
err_impl = np.sum(err_impl_time) / Nt1

err_expl_time = np.abs(hist_expl-baseline_expl)/(Nx_A + Nx_B)
err_expl = np.sum(err_expl_time) / Nt2

print("Implicit error: ", err_impl)
print("Explicit error: ", err_expl)


print(err_impl_time.shape)
# Plot in one plot errors ovver time
plt.figure(figsize=(10, 6))
plt.plot(err_impl_time, label="Implicit Error")
plt.plot(err_expl_time, label="Explicit Error")
plt.xlabel("Time Step")
plt.ylabel("Error")
plt.title("Error Over Time for Implicit and Explicit Methods")
plt.legend()
plt.grid()
plt.savefig("error_plot.png")