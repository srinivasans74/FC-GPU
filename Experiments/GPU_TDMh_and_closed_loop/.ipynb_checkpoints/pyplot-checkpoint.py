import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Read float data from text files
def read_data(filepath):
    with open(filepath, 'r') as f:
        return [float(line.strip()) for line in f if line.strip()]

# Main plotting function
def plot_closed_loop(setpoints, task_labels):
    num_tasks = len(setpoints)
    rtr_files = [f'logs/s{i+1}{i+1}.txt' for i in range(num_tasks)]
    miss_files = [f'logs/deadlinemisst{i+1}.txt' for i in range(num_tasks)]

    # Load data
    rtrs = [read_data(f) for f in rtr_files]
    misses = [read_data(f) for f in miss_files]

    idx = slice(0, 50)  # First 50 control periods

    # Styling
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial']
    })

    fig, axs = plt.subplots(1, 2, figsize=(8, 3), sharex=False)

    # --- Subplot 1: Closed-loop RTR ---
    for i in range(num_tasks):
        sliced_rtr = rtrs[i][idx]
        axs[0].plot(sliced_rtr, linewidth=2, label=task_labels[i])
        axs[0].plot([setpoints[i]] * len(sliced_rtr), '--', linewidth=2)

    axs[0].set_ylabel('RTR', fontsize=13)
    axs[0].set_xlabel('Control Period\n(a) Closed-loop RTR', fontsize=12)
    axs[0].set_ylim([0, 2])
    axs[0].grid(True)
    axs[0].spines['top'].set_visible(False)
    axs[0].legend(fontsize=11)

    # --- Subplot 2: Deadline Misses ---
    for i in range(num_tasks):
        axs[1].plot(np.array(misses[i])[idx], linewidth=2, label=task_labels[i])

    axs[1].set_ylabel('Deadline Miss (%)', fontsize=13)
    axs[1].set_xlabel('Control Period\n(b) Deadline Misses', fontsize=12)
    axs[1].grid(True)
    axs[1].spines['top'].set_visible(False)
    axs[1].legend(fontsize=11)

    for ax in axs:
        ax.tick_params(axis='both', labelsize=12)
        ax.spines['right'].set_visible(False)

    # Save to file
    os.makedirs("Figures", exist_ok=True)
    sep_str = "_".join(str(s) for s in setpoints)
    plt.tight_layout()
    plt.savefig(f'Figures/closed_loop_rtr_miss_{sep_str}.pdf', dpi=300, bbox_inches='tight')
    plt.show()

# Entry point
if __name__ == "__main__":
    args = sys.argv[1:]
    num_args = len(args)

    if num_args < 2 or num_args > 4:
        print("Usage: python plot_closed_loop.py <sep1> <sep2> [<sep3> <sep4>]")
        sys.exit(1)

    setpoints = [float(arg) for arg in args]
    task_labels = [f'Task {i+1}' for i in range(len(setpoints))]

    plot_closed_loop(setpoints, task_labels)