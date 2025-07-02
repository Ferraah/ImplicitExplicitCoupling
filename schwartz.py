import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

def heat_eq_schwarz_interface():
    # Physical and numerical parameters
    L = 100              
    D = 10                 
    T_len = 100
    nx = 200
    dt = 1/10                
    nt = int(T_len / dt)
    dx = L / (nx - 1)
    r = D * dt / dx**2

    # Grid and initial condition
    x_km = np.linspace(0, L, nx) 


    # Subdomain splitting
    mid = nx // 2
    subdomains = [(0, mid), (mid, nx)]
    interface = mid
    
    # 30 degree on first half and 0 on the second
    T = np.zeros(nx)
    T[:mid] = 30  # 30 degrees in the first half
    T[mid:] = 0   # 0 degrees in the second half

    # Snapshot setup
    snapshot_indices = np.linspace(0, nt, 15, dtype=int)
    snapshots = []

    schwarz_iterations = [] 
    max_schwarz_iter = 10
    schwarz_tol = 1e-6

    # Time stepping
    for n in range(1, nt + 1):
        T_prev = T.copy()
        T_curr = T_prev.copy()

        # Schwarz iterations
        for it in range(max_schwarz_iter):
            T_prev_iter = T_curr.copy()
            for i, (start, end) in enumerate(subdomains):
                size = end - start
                diag = [np.ones(size) * (1 + 2*r), np.ones(size-1) * -r, np.ones(size-1) * -r]
                A = diags(diag, [0, -1, 1], format='csr')
                b = T_prev[start:end].copy()

                if i == 0:
                    A[0, 0] += r  # Neumann BC
                    b[0] += 0     # Zero flux
                    b[-1] += r * T_curr[interface]
                else:
                    b[0] += r * T_curr[interface - 1]
                    A[-1, -1] += r
                    b[-1] += 0     # Zero flux

                T_sub = spsolve(A, b)
                T_curr[start:end] = T_sub

            # Compute interface discrepancy
            interface_diff = np.abs(T_curr[interface] - T_curr[interface - 1])
            if interface_diff < schwarz_tol:
                break

        T = T_curr
        schwarz_iterations.append(it)
        print(it)
        if n in snapshot_indices:
            snapshots.append(T.copy())

    # Plot snapshots
    plt.figure(figsize=(10, 6))
    for i, snap in enumerate(snapshots):
        plt.plot(x_km, snap, label=f"t={i * T_len / (len(snapshots)-1):.2f} yrs")
    plt.xlabel("Distance from Equator (km)")
    plt.ylabel("Temperature")
    plt.title("Temperature Evolution Over Time")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("heat_eq_snapshots.png")
    print("Saved 'heat_eq_snapshots.png'")

if __name__ == "__main__":
    heat_eq_schwarz_interface()
