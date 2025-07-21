import numpy as np
import matplotlib.pyplot as plt

# Helper to read float values from file
def read_data(file_path):
    with open(file_path, 'r') as f:
        return [float(line.strip()) for line in f if line.strip()]

# Load GPU-TDMh data
open_s1 = read_data('tdmalogs1/rtr1.txt')
open_s2 = read_data('tdmalogs1/rtr2.txt')
open_p1 = read_data('tdmalogs1/misses1.txt')
open_p2 = read_data('tdmalogs1/misses2.txt')

# Configuration
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial']
})

# Plot indices
idx = slice(1, 50)
setpoint_value = 0.90
setpoint = [setpoint_value] * len(open_s1)

# Setup subplots
fig, axs = plt.subplots(1, 2, figsize=(8, 3))

# --- Subplot (a): RTR ---
axs[0].plot(open_s1[idx], '-', linewidth=2, label='mm1')
axs[0].plot(open_s2[idx], '-.', linewidth=2, label='mm2')
axs[0].plot(np.array(setpoint)[idx], '--', linewidth=2, label='Set Point')
axs[0].set_ylabel('RTR', fontsize=13)
axs[0].set_xlabel('Control Period\n\n(a) GPU-TDMh', fontsize=12)
axs[0].set_ylim([0, 2])
axs[0].tick_params(axis='both', labelsize=12)
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)
axs[0].legend(fontsize=11)

# --- Subplot (b): Deadline Misses ---
axs[1].plot(open_p1[idx], '-', linewidth=2, label='mm1')
axs[1].plot(open_p2[idx], '-.', linewidth=2, label='mm2')
axs[1].set_ylabel('Deadline misses (%)', fontsize=14)
axs[1].set_xlabel('Control Period\n\n(b) GPU-TDMh', fontsize=12)
axs[1].tick_params(axis='both', labelsize=12)
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)
axs[1].legend(fontsize=11)

# Final layout and save
plt.tight_layout()
plt.savefig('Figures/GPUTDMh_openloop_only.pdf', dpi=300, bbox_inches='tight')
plt.show()