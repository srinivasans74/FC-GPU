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

To reproduce **Figure 6a** and **6b**:

1.  Navigate to the `closed-loop` directory:
    ```bash
    cd closed-loop
    ```
2.  Run the experiment script:
    ```bash
    ./experiment.sh
    ```
3.  To visualize the results, run all cells in the `closed-loop.ipynb` Jupyter notebook.

-----


## GPU-TDMH and closed-loop Experiment

This experiment compares closed-loop and FC-GPU, producing **Figure 7a**, **7b**, **7c**, and **7d**. 

1.  Navigate to the `GPU_TDMh_and_closed_loop` directory:
    ```bash
    cd GPU_TDMh_and_closed_loop
    ```
2.  Run the master experiment script:
    ```bash
    ./masterexperiment
    ```
3.  To visualize the plots, open `plot.ipynb` and run all cells. Saved PDF figures are also available in the `Figures/` directory.
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
4. For the Ad-hoc we have choosen three stepsizes a,b,c and they are currently adjusted at runtime when launching them.
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
3.  To visualize the plots, open `one plot.ipynb` and run all cells. Saved PDF figures are also available in the `Figures/` directory. Here we have tuned the gain using matlab. The gains can be adjusted

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

**Note:** More Online experiments and experiments focusing on AMD-GPU will be released in the next version (v.1). 