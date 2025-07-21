# closed-loop Experiments

This section provides a reference implementation ofclosed-loop  (Feedback-Controlled GPU scheduling), as presented in our EMSOFT 2025 paper. Three executables cooperate via POSIX shared memory. The tasks allocate their working sets using zero-copy pinned memory (via `cudaHostAllocMapped` and `cudaHostGetDevicePointer`), allowing the GPU to directly read from and write to the same system RAM that the CPU accesses, thereby eliminating the need for explicit memory copies.

### Building

To compile all programs (which produces `mm`, `stencil`, and `server` executables in the `./bin` directory):

```bash
./exprun.sh
```

This script is designed to perform a clean build automatically whenever source files or Makefiles are modified.

### Running aclosed-loop  Experiment

To launch aclosed-loop  experiment:

1.  **Remove previous logs:**
    ```bash
    rm -rf logs/
    ```
2.  **Launch the experiment:**
    ```bash
    ./experiment.sh
    ```

During execution,closed-loop  runs within the same shell, recording Real-Time Response (RTR) and adapting the period every 4 seconds using a scalar $K\_p$ gain. All metrics are appended to files within the `logs/` directory. Here the gains are small since the scheduler's release times are in ms

### Visualizing Results from a 2-Task MIMO Experiment

After the experiment has finished:

1.  Open the `closedloop.ipynb` Jupyter Notebook.
2.  Run all cells within the notebook to visualize the RTR and other relevant metrics.