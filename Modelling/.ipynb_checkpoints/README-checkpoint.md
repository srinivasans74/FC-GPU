# FC-GPU: Feedback-Controlled GPU Scheduling

## Experiment: Response Time vs Number of Concurrent Tasks (Figure 2)

This experiment replicates **Figure 2** from the paper:  
**"FC-GPU: Feedback Control GPU Scheduling for Real-time Embedded Systems"**

It evaluates how the response time of GPU tasks varies with the number of concurrent executions, illustrating the effect of GPU time-slicing and contention.

## Repository Structure

- `logs/` — Intermediate logs (created automatically)
- `t1` — CUDA executable (compiled from releaseguard1.cu)
- `releaseguard1.cu` — CUDA source file (matrix multiplication kernel)
- `run_process.sh` — Main experiment script
- `response_times.log` — Output log with response times
- `plot.py` — Script to parse log and generate responsetime.pdf
- `responsetime.pdf` — Output plot
- `README.md` — This file

## Experiment Description

- **Purpose**: Measure how response time increases as more GPU tasks are scheduled concurrently.
- **Model Used**: Matrix multiplication using CUDA, run multiple times in parallel.
- **Assumption**: The GPU employs time-slicing for concurrent kernel execution.

### GPU Setup

- CUDA zero-copy memory is used (via `cudaHostAllocMapped`)
- Each task performs 512×512 matrix multiplication
- Timing is measured using CUDA events (`cudaEvent_t`)
- Logs response time per kernel

## How to Run

### Requirements

- CUDA Toolkit
- NVIDIA GPU with compute capability ≥ 3.0
- Python 3

### Python Dependencies

Install required packages using:

```bash
pip install matplotlib numpy
```

### Step 1: Compile the CUDA program

```bash
nvcc -o t1 releaseguard1.cu
```

### Step 2: Run the experiment

```bash
./run_process.sh
```

This script will:
- Run the kernel for 1 to 4 tasks in parallel
- Save all response times to `response_times.log`
- Automatically plot the results using `plot.py`

## Output

- `response_times.log`: Contains response time logs grouped by task count.
- `responsetime.pdf`: A bar chart showing average response time vs number of tasks.

## Plot Description

The Python script `plot.py`:
- Parses `response_times.log`
- Computes average response time for 1, 2, 3, and 4 concurrent tasks
- Plots the results as a PDF
