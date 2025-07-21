# Open-Loop Experiment Guide

This guide explains how to run the Open-loop experiment for the FC-GPU system and visualize the results.

## What is Open-Loop?

In the open-loop setup, there is no feedback control applied. The system does not monitor task response times or adjust task rates dynamically. Each workload runs at a fixed, pre-defined period.

During the experiment, the workload is increased mid-way to simulate a more demanding environment. This allows us to observe how a system without control reacts to sudden changes in load.

Example scenario:
- Total runtime: 100 seconds
- Control period: 4 seconds
- New workload is added at the 13th control period (around 52 seconds)

Initially, one GPU workload is active. At the midpoint, a second workload is introduced. The system continues running both tasks for the remainder of the experiment, still without adjusting rates.

## Running the Experiment

1. Compile the binaries (only if you haven't already or made changes):

```bash
./exprun.sh
```

2. Run the open-loop experiment:

```bash
./experiment.sh
```

This script will automatically execute the experiment and generate logs in the `logs/` directory.

If you want to clear previous results:

```bash
rm -rf logs/
./experiment.sh
```

## Viewing the Plots

To visualize the experiment results:

1. Open the `plot.ipynb` Jupyter Notebook.
2. Run all cells in the notebook.

This will generate plots showing response time behavior across the control periods and highlight the impact of the increased workload during open-loop operation.

## Files

- `exprun.sh`: Compiles the GPU task binaries.
- `experiment.sh`: Launches the open-loop experiment.
- `logs/`: Contains output logs for response times, execution intervals, and other metrics.
- `plot.ipynb`: Notebook for plotting and analysis.
