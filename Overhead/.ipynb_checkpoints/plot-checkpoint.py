import os
import re
import matplotlib.pyplot as plt
import pandas as pd

# Initialize empty lists
tasks = []
meas_means, meas_stds = [], []
ctrl_means, ctrl_stds = [], []
act_means, act_stds = [], []

# Pattern to match files
file_pattern = re.compile(r"mean_times_(\d+)\.txt")

# Parse each matching file
for fname in sorted(os.listdir(".")):
    match = file_pattern.match(fname)
    if not match:
        continue
    with open(fname) as f:
        content = f.read()
        task = int(match.group(1))
        tasks.append(task)

        # Extract mean and std from file
        meas = re.search(r"Meas us\s*: mean=(.*?), std=(.*?)\n", content)
        ctrl = re.search(r"Ctrl us\s*: mean=(.*?), std=(.*?)\n", content)
        act  = re.search(r"Act us\s*: mean=(.*?), std=(.*?)\n", content)

        meas_means.append(float(meas.group(1)))
        meas_stds.append(float(meas.group(2)))

        ctrl_means.append(float(ctrl.group(1)))
        ctrl_stds.append(float(ctrl.group(2)))

        act_means.append(float(act.group(1)))
        act_stds.append(float(act.group(2)))

# Create DataFrame
df = pd.DataFrame({
    "Tasks": tasks,
    "Meas us": meas_means,
    "σMeas": meas_stds,
    "Ctrl us": ctrl_means,
    "σCtrl": ctrl_stds,
    "Act us": act_means,
    "σAct": act_stds
}).sort_values("Tasks")

# Plotting
fig, axs = plt.subplots(1, 3, figsize=(12, 3))

# Measurement Overhead
axs[0].plot(df["Tasks"], df["Meas us"], marker='d', color='green')
axs[0].set_xlabel("Tasks\n\n(a) Response time Measurement", fontsize=14)
axs[0].set_ylabel("Overhead (us)", fontsize=14)
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)
axs[0].tick_params(labelsize=12)

# Controller Overhead
axs[1].plot(df["Tasks"], df["Ctrl us"], marker='*', color='orange')
axs[1].set_xlabel("Tasks\n\n(b) Controller", fontsize=14)
axs[1].set_ylabel("Overhead (us)", fontsize=14)
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)
axs[1].tick_params(labelsize=12)

# Actuator Overhead
axs[2].plot(df["Tasks"], df["Act us"], marker='o', color='blue')
axs[2].set_xlabel("Tasks\n\n(c) Rate adaptation", fontsize=14)
axs[2].set_ylabel("Overhead (us)", fontsize=14)
axs[2].spines['top'].set_visible(False)
axs[2].spines['right'].set_visible(False)
axs[2].tick_params(labelsize=12)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("overhead_updated.pdf")
plt.show()