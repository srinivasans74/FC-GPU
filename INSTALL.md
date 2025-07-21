# INSTALL

## Overview

This repository contains code used to reproduce the main figures in our research paper.

**Directory summary**

| Directory | Purpose | Figure |
|-----------|---------|--------|
| `Overhead/` | Measure runtime overhead of FC‑GPU controller | Figure 5 |
| `Modelling/` | Analytical model and scripts | Figure 3 |
| `Experiments/` | Workloads with FC‑GPU controller integrated | – |

Several experiment variants demonstrate the controller working under different GPU memory configurations (zero‑copy vs. explicit copies) and frequency/parameter settings.

---

## Installation

### 1. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

*Hardware dependencies are listed in `REQUIREMENTS.md`. Alternative hardware may require adjusting controller parameters. A forthcoming update will include MATLAB modules for stability analysis across parameter ranges.*

---

## Reproducing key figures

### Figure 3 – modelling results

```bash
cd Modelling
./run_model.sh
```

Expected console output

```
Generating model analysis...
Figure 3 saved as model_figure3.png
```

---

### Figure 5 – overhead measurement

```bash
cd Overhead
./experiment.sh
```

Expected console output

```
Running overhead experiments...
Figure 5 saved as overhead_figure5.png
```

---

## Experiments directory

The `Experiments/` folder contains all benchmark workloads with the FC‑GPU controller embedded. A standalone scheduler is possible, but embedding lets the workload invoke control routines directly. 

### GPU time‑slicing

* NVIDIA: enable with  
  ```bash
  sudo nvidia‑smi -c 0
  ```
* AMD: run  
  ```bash
  ./amd_timeslice.sh
  ```

### Porting CUDA workloads to AMD ROCm

```bash
hipify-perl *.cu > *.cpp        # convert CUDA (.cu) to HIP (.cpp)
```

Add the above line to `experiment.sh`, then run

```bash
./experiment.sh
```

The FC‑GPU controller will operate identically after conversion.

---

## Notes

* Each script stores its generated figure in the same directory.
* Ensure shell scripts are executable:  
  ```bash
  chmod +x *.sh
  ```
* Additional workloads and experiments will be released in future updates.

---

## Troubleshooting & support

Open an issue in the repository or contact the maintainer listed in `LICENSE` if you encounter problems.
