import matplotlib.pyplot as plt

# Parameters
C = 0.1           # HF = C * (SST - T_air)
alpha = 0.01      # SST change = alpha * HF
T_air = 290.0     # Constant
coupling_steps = 1000
substeps = 6      # per coupling period

# --- PARALLEL ALGORITHM (STAGGERED) ---
SST_parallel = [289.0]  # Initial SST
HF_parallel = [C * (SST_parallel[0] - T_air)]  # Initial HF

for step in range(coupling_steps):
    sst_current = SST_parallel[-1]
    hf_lagged = HF_parallel[-1]  # Uses previous flux

    # Ocean substeps with fixed HF
    for _ in range(substeps):
        sst_current += alpha * hf_lagged
    SST_parallel.append(sst_current)

    # Atmosphere computes HF with current SST
    hf_new = C * (sst_current - T_air)
    HF_parallel.append(hf_new)

# --- SCHWARZ ITERATIVE METHOD ---
SST_schwarz = [289.0]
HF_schwarz = [C * (SST_schwarz[0] - T_air)]
n_iter = 1000

for step in range(coupling_steps):
    sst_iter = [SST_schwarz[-1]]
    hf_iter = [HF_schwarz[-1]]

    for k in range(n_iter):
        # Atmosphere computes HF using latest SST
        hf_k = C * (sst_iter[-1] - T_air)
        hf_iter.append(hf_k)

        # Ocean updates SST starting from initial state
        sst_k = sst_iter[0]
        for _ in range(substeps):
            sst_k += alpha * hf_k
        sst_iter.append(sst_k)

    SST_schwarz.append(sst_iter[-1])
    HF_schwarz.append(hf_iter[-1])

# Provide numerical comparison at each step
comparison = {
    'Step': list(range(len(SST_parallel))),
    'SST_Parallel': SST_parallel,
    'SST_Schwarz': SST_schwarz,
    'HF_Parallel': HF_parallel,
    'HF_Schwarz': HF_schwarz,
}
print(comparison)

# Plot comparison
plt.figure(figsize=(10, 5))
plt.plot(SST_parallel, 'o-', label='SST - Parallel')
plt.plot(SST_schwarz, 's-', label='SST - Schwarz')
plt.plot(HF_parallel, 'o--', label='HF - Parallel')
plt.plot(HF_schwarz, 's--', label='HF - Schwarz')
plt.xlabel('Coupling Step')
plt.ylabel('Value')
plt.title('Comparison: Parallel vs Schwarz Coupling')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

