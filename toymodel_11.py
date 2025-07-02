import numpy as np
import matplotlib.pyplot as plt

# =========================
# Parameters & Setup
# =========================
L_A, L_B = 1.0, 1.0
Nx_A, Nx_B = 50, 50
alpha_A, alpha_B = 1.16e-4, 1.16e-4  # Same thermal diffusivity for both models
dx = L_A / (Nx_A - 1)
dt = 0.4 * dx**2 / max(alpha_A, alpha_B)

# =========================
# Utility Functions
# =========================
def make_plot_1d(history, file_name):
    plt.figure(figsize=(10, 6))
    plt.plot(history)
    plt.xlabel("Spatial Index")
    plt.grid()
    plt.savefig(file_name)
    plt.close()

def make_plot(full_history, num_snapshots, file_name):
    plt.figure(figsize=(10, 6))
    step = max(1, len(full_history)//num_snapshots)
    for i, snapshot in enumerate(full_history[::step]):
        plt.plot(snapshot, label=f"Snapshot {i}")
    plt.xlabel("Spatial Index")
    plt.ylabel("Temperature")
    plt.title("Temperature Evolution Over Time")
    plt.legend()
    plt.grid()
    plt.savefig(file_name)
    plt.close()

def initial_conditions():
    T_A = np.linspace(10.0, 20.0, Nx_A)
    T_B = np.linspace(10.0, 20.0, Nx_B)
    return T_A, T_B

# =========================
# Numerical Methods
# =========================
def update_heat_explicit(T, alpha, dx, dt, n):
    T_new = T.copy()
    T_new[1:-1] = T[1:-1] + alpha * dt / dx**2 * (T[:-2] - 2*T[1:-1] + T[2:])
    T_new[0] = T_new[1]       # Neumann BC left
    T_new[-1] = T_new[-2]     # Neumann BC right
    return T_new

# =========================
# Coupling Schemes
# =========================
def monolithic_coupling(Nt):
    T_A, T_B = initial_conditions()
    T_full = np.hstack((T_A, T_B))
    full_history = []
    for n in range(Nt):
        T_full = update_heat_explicit(T_full, alpha_A, dx, dt, n)
        full_history.append(T_full.copy())
    print("Timesteps number:", Nt)
    print("Total model runs:", Nt)
    make_plot(full_history, 5, "output_monolithic.png")
    return full_history

def explicit_coupling(Nt, n_cpl):
    T_A, T_B = initial_conditions()
    full_history = []
    models_run = 0

    for ts in range(Nt):

        if ts % n_cpl == 0:
            T_if = 0.5 * (T_A[-1] + T_B[0])
            T_A[-1] = T_if
            T_B[0] = T_if
        T_A = update_heat_explicit(T_A, alpha_A, dx, dt, ts)
        T_B = update_heat_explicit(T_B, alpha_B, dx, dt, ts)
        models_run += 1

        full_history.append(np.hstack((T_A, T_B)))

    print("Timesteps number:", Nt)
    print("Total model runs:", models_run)
    make_plot(full_history, 5, "output_explicit.png")
    return full_history

def implicit_coupling(Nt, n_cpl, max_iter):
    n_sloops = Nt // n_cpl

    T_A, T_B = initial_conditions()
    full_history = []
    models_run = 0

    for i_sloop in range(n_sloops):

        # State fields at the beginning of the Schwartz windows
        T_A_save = T_A.copy()
        T_B_save = T_B.copy()

        for k_swrz in range(max_iter):

            T_A_prev = T_A_save.copy()
            T_B_prev = T_B_save.copy()

            # Apply interface conditions given the previous state
            T_if = 0.5 * (T_A_prev[-1] + T_A_prev[0])
            T_A_prev[-1] = T_if
            T_B_prev[0] = T_if

            for ts in range(n_cpl):
                T_A = update_heat_explicit(T_A_prev, alpha_A, dx, dt, i_sloop * n_cpl + ts)
                T_B = update_heat_explicit(T_B_prev, alpha_B, dx, dt, i_sloop * n_cpl + ts)
                models_run += 1
                err = (np.sum(np.abs(T_A_prev - T_A)) + np.sum(np.abs(T_B_prev- T_B))) / (Nx_A + Nx_B)
                T_A_prev = T_A.copy()
                T_B_prev = T_B.copy()
                #print(f"Schwarz iteration {k_swrz}, step {ts}, error: {err:.6f}")

        full_history.append(np.hstack((T_A, T_B)))

    print("Timesteps number:", Nt)
    print("Total model runs:", models_run)
    make_plot(full_history, 5, "output_schwarz.png")
    return full_history

# =========================
# Main Simulation
# =========================
def main():
    simulation_time = 1000
    cpl_frequency = 10
    Nt = int(simulation_time // dt)
    Nt = (Nt // cpl_frequency) * cpl_frequency

    print("Running baseline")
    baseline = np.asarray(monolithic_coupling(Nt), dtype=object)

    print("\nRunning implicit")
    hist_impl = np.asarray(implicit_coupling(Nt, cpl_frequency, 10), dtype=object)

    print("\nRunning explicit")
    hist_expl = np.asarray(explicit_coupling(Nt, cpl_frequency), dtype=object)

    print("\nhist_impl shape:", hist_impl.shape)
    print("hist_expl shape:", hist_expl.shape)
    err_impl = np.sum(np.abs(hist_impl - baseline[::cpl_frequency])) / (Nt/cpl_frequency * (Nx_A + Nx_B))
    err_expl = np.sum(np.abs(hist_expl - baseline[::cpl_frequency])) / (Nt/cpl_frequency * (Nx_A + Nx_B))
    # print(f"Implicit error: {err_impl:.4f}")
    # print(f"Explicit error: {err_expl:.4f}")

if __name__ == "__main__":
    main()
