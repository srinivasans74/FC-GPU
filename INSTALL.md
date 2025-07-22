# INSTALL

## Overview

This repository contains code used to reproduce the main figures in our research paper.

### Directory Summary

| Directory       | Purpose                                            | Figure     |
|----------------|----------------------------------------------------|-----------  |
| `Overhead/`     | Measure runtime overhead of the FC-GPU controller | Figure 5    |
| `Modelling/`    | Analytical model and scripts                      | Figure 3    |
| `Experiments/`  | Workloads with the FC-GPU controller integrated   | Figures 6-11|

Different experiment variants demonstrate how the controller performs under various GPU memory configurations (zero-copy vs. explicit copies) and different control parameters.


---

## Installation

### 1. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

Hardware requirements are listed in `REQUIREMENTS.md`. Some parameter tuning might be needed for different GPUs. MATLAB modules for stability analysis will be released in an upcoming update.


---

## GPU Time-Slicing Setup

### NVIDIA

Enable time-slicing:
```bash
sudo nvidia-smi -c 0
```

### AMD

Run:
```bash
./amd_timeslice.sh
```

---


## Reproducing Key Figures

### Figure 3 – Modelling Results

```bash
cd Modelling
./run_model.sh
```

This will generate and save `model_figure3.png`.

---

### Figure 5 – Overhead Measurement

```bash
cd Overhead
./experiment.sh
```

This will generate and save `overhead_figure5.png`.

---

## Figure 6-11

The `Experiments/` folder contains all benchmark workloads with the FC-GPU controller embedded. To run these:

1. Navigate to the directory:
    ```bash
    cd Experiments
    ```

2. Follow the steps in the provided `README.md`.

The results should be reproducible as shown in the paper.


## Porting CUDA Workloads to AMD ROCm

To convert CUDA code to HIP for AMD GPUs Please check (Experiments/FC-GPU_amd/) [`AMD`](Experiments/FC-GPU_amd/) please run the perl script used there to port the code. Note for later versions HiphostAlloc may be depreceated. If yes please change it to hipHostMalloc.


## Important Note on GPU Tuning

**If you are using a GPU different from the one used for initial development and tuning, the controller gains (`a11` through `a44`) will likely need to be re-tuned.** These gains are critical for the stability of the control system and are highly dependent on the underlying hardware's performance characteristics.

We recommend performing a stability analysis (as per your system design) to determine the optimal gain values for your specific GPU to ensure the controller operates effectively and maintains real-time guarantees. Failure to re-tune these gains on different hardware may lead to performance degradation or instability. In addition Please ensure the system has large RAM to ensure optimal performance and avoid out-of-memory issues, especially when utilizing page-locked (pinned) memory for data transfer, the system running this code requires a large amount of available RAM. This is critical as functions like cudaHostAlloc (or hipHostMalloc for HIP) allocate memory that cannot be swapped to disk.


## Support

For help, open an issue in the repository or contact the maintainer listed in the `LICENSE` file.
