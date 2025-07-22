# System Requirements

This document describes the system-level configuration required to build, run, and evaluate the FC-GPU artifact. The artifact supports both AMD ROCm and NVIDIA CUDA-based systems.

## Operating System and Kernel

- OS: Ubuntu 20.04.6 LTS
- Linux Kernel: 5.15.0-139-generic
- Architecture: x86_64

## Toolchain

- GCC version: 9.4.0
- CMake version: 3.19.6
- glibc version: 2.31
- make version: 4.2.1
- pthreads: Built-in via glibc

## GPU Acceleration Support

### AMD ROCm Setup

- ROCm version: 5.6.0
- GPU: AMD Instinct MI100
- ROCm runtime components: HIP, LLVM, SMI
- ROCm libraries available in `/opt/rocm/lib`:
  - hipBLAS
  - rocBLAS
  - rocFFT
  - rocRAND
  - rocSOLVER
  - rocSPARSE
  - RCCL

Kernel modules that should be loaded:

- amdgpu
- amdttm
- amdkcl
- amddrm_ttm_helper
- drm_kms_helper

Environment variables to be set:

```
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm/hip
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib
```

### Enabling Time-Slicing on AMD GPUs

By default, AMD GPUs use spatial sharing. We configure the AMD GPU driver to run in single-process time-slicing mode to prioritize one process at a time.

This is achieved by modifying the `amdgpu` driver parameter:

```
hws_max_conc_proc=1
```

Instructions:

1. Make the script executable:
   ```
   chmod +x amd_timeslice.sh
   ```
2. Run the script:
   ```
   ./amd_timeslice.sh
   ```

This will recompile and reload the driver with the time-slicing setting enabled.

### NVIDIA CUDA Setup (Optional / Alternative)

- NVIDIA Driver version: 560.35.03
- CUDA Toolkit version: 11.6.124
- cuDNN version: Not explicitly checked (assumed installed via system package manager)
- GPU: Use `nvidia-smi` to confirm specific model if needed

Validation commands:

```
nvcc --version
nvidia-smi
```

## Python Environment

A complete list of Python libraries and versions is provided in the `requirements.txt` file included in this repository.

To install the Python environment:

```
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The `requirements.txt` file captures all libraries used, including those needed for evaluation, plotting, and runtime execution.

## Notes

- The artifact has been tested and validated on both AMD ROCm 5.6 and NVIDIA CUDA 11.6 environments.
- CPU fallback may work for some components but GPU acceleration is recommended for best performance.
- To ensure full reproducibility, please match or exceed the versions listed above for your system.
- In addition Please ensure the system has large RAM to ensure optimal performance and avoid out-of-memory issues, especially when utilizing page-locked (pinned) memory for data transfer, the system running this code requires a large amount of available RAM. This is critical as functions like cudaHostAlloc (or hipHostMalloc for HIP) allocate memory that cannot be swapped to disk.
