# INSTALL

## Overview

This repository contains code used to reproduce the main figures in our research paper.

### Directory Summary

| Directory       | Purpose                                            | Figure    |
|----------------|----------------------------------------------------|-----------|
| `Overhead/`     | Measure runtime overhead of the FC-GPU controller | Figure 5  |
| `Modelling/`    | Analytical model and scripts                      | Figure 3  |
| `Experiments/`  | Workloads with the FC-GPU controller integrated    | –         |

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

## Experiments

The `Experiments/` folder contains all benchmark workloads with the FC-GPU controller embedded. To run these:

1. Navigate to the directory:
    ```bash
    cd Experiments
    ```

2. Follow the steps in the provided `README.md`.

The results should be reproducible as shown in the paper.


## Porting CUDA Workloads to AMD ROCm

To convert CUDA code to HIP for AMD GPUs:

```bash
for file in *.cu; do
    hipify-perl "$file" > "${file%.cu}.cpp"
done
hipcc *.cpp -o executable
```
Include this in `experiment.sh` and then run:

```bash
./experiment.sh
```

The FC-GPU controller remains functional post-porting. Users may need to tune control gains slightly.

---

## Notes

- Figures will be saved in their respective directories.
- Ensure all scripts are executable:
  ```bash
  chmod +x *.sh
  ```
- Additional workloads and experimental scenarios will be added in future versions.

---

## Support

For help, open an issue in the repository or contact the maintainer listed in the `LICENSE` file.
