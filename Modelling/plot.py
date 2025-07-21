import matplotlib.pyplot as plt
import numpy as np
import re

log_file = "response_times.log"
response_times = {}

task_count = None  

with open(log_file, 'r') as f:
    lines = f.readlines()

for line in lines:
    if "Running with" in line:
        match = re.search(r'Running with (\d+)', line)
        if match:
            task_count = int(match.group(1))
            response_times[task_count] = []
    elif "Response time" in line and task_count is not None:
        match = re.search(r"Response time = ([\d.]+)", line)
        if match:
            response_times[task_count].append(float(match.group(1)))

# Compute average response times
tasks = sorted(response_times.keys())
execution_times = [np.mean(response_times[t]) for t in tasks]

# Plotting
colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightsalmon']
hatches = ['/', '++', 'xx', 'oo']

fig, ax = plt.subplots(figsize=(2.5, 2.3))
bars = ax.bar(tasks, execution_times, color=colors[:len(tasks)], edgecolor='black')

for bar, hatch in zip(bars, hatches):
    bar.set_hatch(hatch)

ax.set_xlabel("Number of Tasks", fontsize=12, fontweight='bold')
ax.set_ylabel("Response Time (ms)", fontsize=12, fontweight='bold')
ax.set_xticks(tasks)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('responsetime.pdf')
plt.show()