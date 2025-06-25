import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

def solve_heat_equation_schwarz():
    # Parameters
    L = 1.0           # Length of the domain
    T = 0.1           # Total simulation time
    alpha = 3    # Thermal diffusivity
    nx = 100          # Total number of grid points
    nt = 500          # Number of time steps
    dt = T / nt       # Time step size
    dx = L / (nx - 1) # Spatial step size
    num_snapshots = 20
    # Domain decomposition parameters
    num_subdomains = 2
    mid_point = nx // num_subdomains
    interface_idx = mid_point  # Interface location
    
    # Initial condition (rectangular pulse)
    u = np.zeros(nx)
    # u[int(0.3/dx):int(0.7/dx)] = 1.0
    u[0:49] = 1.0
    
    # Boundary conditions (Dirichlet)
    left_bc = 0.0
    right_bc = 0.0
    
    # Create the grid
    x = np.linspace(0, L, nx)
    
    # Define non-overlapping subdomains
    subdomains = [(0, mid_point), (mid_point, nx)]
    
    # Coefficient for finite difference scheme
    r = alpha * dt / dx**2
    
    # Convergence parameters for Schwarz iterations
    max_schwarz_iters = 20     # Maximum iterations per time step
    tolerance = 1e-5           # Convergence tolerance
    schwarz_iter_counts = []    # Track iteration counts
    interface_history = []      # Track interface values
    
    # Snapshots for visualization
    snapshots = [u]
    snapshot_times = np.linspace(0, nt, num_snapshots, dtype=int)
    
    # Time-stepping loop
    for n in range(1, nt+1):
        u_prev = u.copy()  # Solution at previous time step
        
        # Schwarz iteration control
        schwarz_iter = 0
        converged = False
        u_current = u_prev.copy()  # Start with solution from previous time step
        
        # Store initial interface values
        prev_interface_left = u_current[interface_idx - 1]  # Left domain's right boundary
        prev_interface_right = u_current[interface_idx]      # Right domain's left boundary
        
        while not converged and schwarz_iter < max_schwarz_iters:
            # Process each subdomain sequentially (multiplicative Schwarz)
            for i, (start, end) in enumerate(subdomains):
                # Subdomain size
                sub_nx = end - start
                
                # Create finite difference matrix
                diagonals = [np.ones(sub_nx) * (1 + 2*r), 
                            np.ones(sub_nx-1) * -r, 
                            np.ones(sub_nx-1) * -r]
                A = diags(diagonals, [0, -1, 1], format='csr')
                
                # Right-hand side vector
                b = u_prev[start:end].copy()
                
                # Apply boundary conditions
                if i == 0:  # Left subdomain
                    # Left boundary (global)
                    b[0] += r * left_bc
                    # Right boundary (interface)
                    b[-1] += r * u_current[interface_idx]
                else:  # Right subdomain
                    # Left boundary (interface)
                    b[0] += r * u_current[interface_idx - 1]
                    # Right boundary (global)
                    b[-1] += r * right_bc
                
                # Solve the system for this subdomain
                u_sub = spsolve(A, b)
                
                # Update solution
                u_current[start:end] = u_sub
            
            # Save current interface values
            curr_interface_left = u_current[interface_idx - 1]
            curr_interface_right = u_current[interface_idx]
            
            # Calculate interface changes
            delta_left = abs(curr_interface_left - prev_interface_left)
            delta_right = abs(curr_interface_right - prev_interface_right)
            max_interface_delta = max(delta_left, delta_right)
            
            # Update for next iteration
            prev_interface_left = curr_interface_left
            prev_interface_right = curr_interface_right
            schwarz_iter += 1

            # Check convergence (interface-based criterion)
            if max_interface_delta < tolerance:
                converged = True
        
        # Update global solution
        u = u_current
        schwarz_iter_counts.append(schwarz_iter)
        interface_history.append((curr_interface_left, curr_interface_right))

        if n in snapshot_times:
            snapshots.append(u.copy())

        # Progress monitoring
        if n % 50 == 0:
            status = "✓" if converged else f"max iters ({max_schwarz_iters})"
            print(f"Time step {n:4d}/{nt}: {schwarz_iter} Schwarz iters {status}")
            print(f"  Interface values: {curr_interface_left:.6f}, {curr_interface_right:.6f}")
            print(f"  Max delta: {max_interface_delta:.2e}")

     
    # Plot interface evolution
    plt.figure(figsize=(10, 6))
    left_vals = [val[0] for val in interface_history]
    right_vals = [val[1] for val in interface_history]
    plt.plot(left_vals, 'r-', label='Left Domain Interface')
    plt.plot(right_vals, 'b--', label='Right Domain Interface')
    plt.xlabel('Time Step')
    plt.ylabel('Interface Value')
    plt.title('Interface Value Evolution')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("interface_evolution.png")
    
    # Plot iteration convergence
    plt.figure(figsize=(10, 6))
    plt.plot(range(nt), schwarz_iter_counts, 'g-')
    plt.xlabel('Time Step')
    plt.ylabel('Schwarz Iterations')
    plt.title('Convergence Behavior Over Time')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("convergence_history.png")
    
    # Plot snapshots
    plt.figure(figsize=(10, 6))
    for i, snap in enumerate(snapshots):
        plt.plot(x, snap, label=f"Step {snapshot_times[i]}")
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('Snapshots of 1D Heat Equation with Schwarz Coupling')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("snapshots.png")

if __name__ == "__main__":
    solve_heat_equation_schwarz()