# FC-GPU: Overhead Measurement Experiment

This experiment measures the runtime overhead introduced by the FC-GPU feedback controller. It replicates the results shown in **Figure 5** of our paper, reporting the overheads associated with:

- **(a)** Response Time Monitoring  
- **(b)** Controller Computation  
- **(c)** Rate Adaptation (Actuation)  

All overheads are reported in microseconds (µs).

---

## Directory Contents

- `overhead.cpp` – C++ implementation of the timing experiment with full feedback loop
- `experiment.sh` – Script to compile and run the experiment
- `mean_times_*.txt` – Output files containing measured overheads for each task count
- `plot_overhead.py` – Python script to generate subplots for each overhead component
- `overhead_updated.pdf` – The combined figure (same layout as in the paper)

---

## System Setup

This experiment is CPU-only and was tested on:

- **Processor**: Intel Core i5-12500 (12 threads, 6 cores, no hyper-threading required)
- **Operating System**: Ubuntu 20.04.6 LTS (x86_64)
- **Kernel**: Linux 5.15.0-139-generic
- **Compiler**: `g++-9` with `-std=c++17`
- **Python**: Version 3.8 or above with `matplotlib` and `pandas`

> The experiment does **not** require a GPU.

---

## How to Run the Experiment

1. **Compile and Run Controller**  
   Run the experiment shell script which compiles and executes the controller for task counts from 2 to 16:

   ```bash
   chmod +x experiment.sh
   ./experiment.sh