# GPU-TDMh vs Closed-Loop Control (FC-GPU) for GPU Scheduling

This repository provides an experimental setup to compare an **open-loop GPU scheduler (GPU-TDMh)** with a **closed-loop feedback-controlled GPU scheduling system (FC-GPU)** using CUDA workloads.

## Objective

The goal of this experiment is to demonstrate the necessity of **closed-loop control** in GPU scheduling. It highlights how **open-loop** approaches like GPU-TDMh can fail under dynamic load conditions, while **closed-loop control** adapts execution periods in real-time to meet system-level goals.

## Background

### GPU-TDMh (Open-Loop)
- Static time-slicing using offline profiling
- Fixed GPU slice sizes per task
- No runtime feedback or adaptation
- Implemented in: `mmslice11_fixed.cu`, `mmslice22_fixed.cu`
- Plotted via: `gputdhmhplot.py`

### FC-GPU (Closed-Loop Control)
- Feedback controller adjusts periods dynamically
- Period modulation via slack-based feedback
- Gain matrix `[a11, a12, a21, a22]` controls response
- Implemented in: `mmnew1.cu`, `mmnew2.cu`
- Plotted via: `pyplot.py`

## Directory Structure

- `runit.py` — Launches FC-GPU experiments  
- `GPUTDMh.sh` — Launches GPU-TDMh baseline  
- `closedloop.sh` — Launches closed-loop controller (FC-GPU)  
- `server.cpp` — Feedback controller (2-task version)  
- `tdhmaserver_fixed.cpp` — GPU-TDMh control logic (if needed)  
- `mmnew1.cu`, `mmnew2.cu` — FC-GPU CUDA kernels  
- `mmslice11_fixed.cu`, `mmslice22_fixed.cu` — GPU-TDMh CUDA kernels  
- `logs/` — All log output (slack, deadline miss, periods)  
- `figures/` — Auto-generated figures  
- `pyplot.py` — FC-GPU plotting  
- `gputdhmhplot.py` — GPU-TDMh plotting  
- `README.md` — This file  

## Running Experiments

### 1. Run Closed-Loop Controller (FC-GPU)

```bash
./closedloop.sh
```

This will:
- Compile `mmnew1.cu`, `mmnew2.cu`, and `server.cpp`
- Start tasks with shared memory
- Run feedback-controlled scheduling
- Automatically generate plots via `pyplot.py`
- Output to `logs/` and `figures/`

### 2. Run Open-Loop Scheduler (GPU-TDMh)

```bash
./GPUTDMh.sh
```

This will:
- Launch the fixed-period GPU-TDMh experiment using `mmslice*.cu`
- Logging to `tdmalogs/` or `tdmalogs1/`
- Plotting is **separate**, using:

```bash
python3 gputdhmhplot.py
```

## Gain Matrix (FC-GPU)

The controller uses a matrix of gains:

```cpp
float a11 = 0.001700 * 1;
float a12 = 0.0001968 * 1;
float a21 = 0.0001968 * 1;
float a22 = 0.001700 * 1;
```

You may tune these based on:
- Worst-case execution time (WCET)
- Desired slack bounds
- System responsiveness and stability

> **Note:** Figure generation is automated but separate for the two systems. Do not combine them as in the paper.

## Key Takeaways

- **GPU-TDMh** is predictable but fragile. It cannot handle changing system conditions.
- **FC-GPU** is adaptive. It:
  - Tracks RTR in real-time
  - Dynamically adjusts periods
  - Maintains system performance even under dynamic interference

## To Run Both Experiments

```bash
./masterexperiment.sh
```

## Reference

This setup is based on the GPU control methodology described in *Section 2 & 5.1* of the referenced paper, comparing TDMh and feedback-controlled GPU scheduling.
