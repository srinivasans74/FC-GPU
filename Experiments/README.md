# Experiment Guide

This guide outlines how to run the experiments and reproduce the figures presented in the paper.
-----
## Open-Loop Experiments

To reproduce **Figure 6a** and **6b**:

1.  Navigate to the `open-loop` directory:
    ```bash
    cd open-loop
    ```
2.  Run the experiment script:
    ```bash
    ./experiment.sh
    ```
3.  To visualize the results, run all cells in the `openloop.ipynb` Jupyter notebook.

-----

## Closed-Loop Experiments

To reproduce **Figure 6c** and **6d**:

1.  Navigate to the `closed-loop/` directory:
    ```bash
    cd closed-loop/
    ```
2.  Run the experiment script:
    ```bash
    ./experiment.sh
    ```
3.  To visualize the results, run all cells in the `closedloop.ipynb` Jupyter notebook.

-----



## GPU-TDMH Experiment

This experiment compares closed-loop and FC-GPU, producing **Figure 7a**, **7b**, **7c**, and **7d**.

1.  Navigate to the `GPU_TDMh` directory:
    ```bash
    cd GPU_TDMh
    ```
2.  Run the master experiment script:
    ```bash
    ./master experiment
    ```
3.  To visualize the plots, open `one plot.ipynb` and run all cells. Saved PDF figures are also available in the `Figures/` directory.
4.  To return to the parent directory:
    ```bash
    cd ..
    ```

-----

## Ad-hoc Experiments

1.  Navigate to the `Ad-hoc/` directory:
    ```bash
    cd Ad-hoc/
    ```
2.  Run the experiment script:
    ```bash
    ./experiment.sh
    ```
3.  To visualize the plots, open `one plot.ipynb` and run all cells. Saved PDF figures are also available in the `Figures/` directory.

-----

## SISO Experiments

1.  Navigate to the `SISO/` directory:
    ```bash
    cd SISO/
    ```
2.  Run the experiment script:
    ```bash
    ./experiment.sh
    ```
3.  To visualize the plots, open `one plot.ipynb` and run all cells. Saved PDF figures are also available in the `Figures/` directory.

-----

## FC-GPU Experiments

To reproduce **Figure 9c** and **9d**:

1.  Navigate to the `FC-GPU/` directory:
    ```bash
    cd FC-GPU/
    ```
2.  Run the experiment script:
    ```bash
    ./experiment.sh
    ```
3.  To visualize the results, run all cells in the `closedloop.ipynb` Jupyter notebook.

-----


## Online Experiments

To reproduce **Figure 11*

1.  Navigate to the `online_new_task_arrival_nvidia` directory:
    ```bash
    cd online_new_task_arrival_nvidia/
    ```
2.  Run the experiment script:
    ```bash
    ./experiment.sh
    ```
3.  To visualize the results, run all cells in the `plot.ipynb` Jupyter notebook.

-----

## AMD GPU

To execute FC-GPU in AMD GPU
cd FC-GPU_amd/
./experiment.sh
cd ..
Note this is for two tasks only we show how the framework is portable here
-----


## To run all the experiments


1.  Run the experiment script:
    ```bash
    ./masterexperiment.sh
    ```
-----


**Note:** Experiment 5.4 (Table 2 and 3) will be released in (v.1.5).


## Important Note on GPU Tuning

**If you are using a GPU different from the one used for initial development and tuning, the controller gains (`a11` through `a44`) will likely need to be re-tuned.** These gains are critical for the stability of the control system and are highly dependent on the underlying hardware's performance characteristics.

We recommend performing a stability analysis (as per your system design) to determine the optimal gain values for your specific GPU to ensure the controller operates effectively and maintains real-time guarantees. Failure to re-tune these gains on different hardware may lead to performance degradation or instability.
