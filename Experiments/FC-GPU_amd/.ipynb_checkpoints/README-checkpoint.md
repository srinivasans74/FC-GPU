# FC-GPU Experiments for AMD GPUs

This repository contains a reference implementation of FC-GPU (Feedback-Controlled GPU scheduling), as presented in our EMSOFT 2025 paper. It demonstrates a system where three cooperating executables share a POSIX shared memory region:

  * **`t1`** — Real-Time GPU Task 1
  * **`t2`** — Real-Time GPU Task 2
  * **`server`** — Central controller

Each GPU task utilizes **zero-copy pinned memory** (via `cudaHostAllocMapped` / `cudaHostGetDevicePointer` in CUDA, or HIP's `hipHostMallocMapped` / `hipHostGetDevicePointer`). This allows the GPU to directly read from and write to the same host RAM the CPU uses, eliminating the need for explicit memory copies and improving performance.

-----

## Prerequisites

To build and run the FC-GPU experiments, you'll need the following:

  * **ROCm (5.6 or newer)** with `hipcc`
  * **C++17 compiler** (e.g., `g++`)
  * **Perl** (for the patch script)
  * **Python 3** with `matplotlib` (for plotting results)

-----

## 1\. One-Step Build & Run

This repository includes `experiment.sh`, a self-contained script that automates the entire process:

  * Cleans shared memory and old log files.
  * Builds the central controller (`server.cpp`).
  * Converts CUDA tasks (`.cu` files) to HIP C++ (`.cpp` files) and compiles them.
  * Applies automatic source patches (see Section 2).
  * Launches the tasks and controller in the correct order.
  * Runs the experiment for a fixed duration.
  * Invokes `pyplot.py` to generate final figures.

To execute the experiment, follow these steps:

```bash
chmod +x experiment.sh
rm -rf logs/
./experiment.sh
```

All numeric metrics, such as response times, periods, slack, and deadline misses, will be saved in the `logs/` directory.

-----

## 2\. Automatic HIP-API Patching

Because HIP's host allocation and many API calls require explicit casts and error checks, we've included `patch_casts_and_wrap.pl`. This Perl filter automatically modifies the source code to ensure compatibility and robust error handling:

1.  Inserts a standard `HIP_CHECK(call)` macro at the top of each `.cpp` file.
2.  Adds the required `(void**)&…` cast to calls like `hipHostAlloc`, `hipHostMalloc`, and `hipHostGetDevicePointer`.
3.  Wraps any single-statement HIP runtime calls (e.g., `hipMemcpy`, `hipMalloc`, `hipEventCreate`) with `HIP_CHECK( … );`.

This patching process ensures:

  * Compile-time errors are eliminated.
  * Any HIP runtime failure aborts immediately with a diagnostic message.

The key macro inserted is:

```c++
#define HIP_CHECK(call)                                                          \
  do {                                                                           \
    hipError_t err = call;                                                       \
    if (err != hipSuccess) {                                                     \
      std::cerr << "HIP Error: "                                                 \
                << hipGetErrorString(err)                                        \
                << " at " << __FILE__                                            \
                << ":" << __LINE__                                               \
                << " in " << #call << std::endl;                                 \
      exit(EXIT_FAILURE);                                                        \
    }                                                                            \
  } while (0)
```

The `patch_casts_and_wrap.pl` script is automatically invoked by `experiment.sh`, so you typically don't need to run it manually.

-----

## 3\. Plotting Results

After the experiment concludes, you can visualize the results using the provided Jupyter Notebook.

1.  Open `FC-GPU.ipynb` in Jupyter.
2.  Run all notebook cells to:
      * Load the CSV files from the `logs/` directory.
      * Plot real-time response, period adaptation, deadline misses, slack, and other relevant metrics.
      * Compare open-loop versus closed-loop behavior.

-----

## 4\. Customization

You can customize various aspects of the experiment:

  * **Setpoints & Periods:** Modify the arrays within `experiment.sh` to adjust task setpoints and periods.
  * **Duration:** Adjust the `duration` variable (in seconds) in `experiment.sh` to change the experiment's runtime.


## 5\. Troubleshooting

Here are some common issues and their solutions:

  * **"undefined reference to main"**: Ensure each task's `.cu` file defines a `main()` function, or add one if missing.
  * **Deprecation warnings for `hipHostAlloc`**: These can be silenced or the calls can be replaced with `hipHostMalloc`.
  * **To rebuild after editing sources or Makefiles**: Simply rerun `./experiment.sh`.

-----
