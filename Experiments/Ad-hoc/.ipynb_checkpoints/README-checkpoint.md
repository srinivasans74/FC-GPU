# Ad-hoc Experiment Guide

This guide explains how to run the Ad-hoc experiments and visualize the results.

### Running the Experiment

To run the Ad-hoc experiment:


1.  **Compilation (if needed):**
    First, ensure the code is compiled using `exprun.sh`. If you've made changes to the source code or are running for the first time, execute:

    ```bash
    ./exprun.sh
    ```

2.  **Adjusting Step Size:**
    Before launching the experiment, you can adjust the step size for the Ad-hoc controller by editing the `experiment.sh` file directly.

3.  **Execute the experiment script:**

    ```bash
    ./experiment.sh
    ```

    This script will generate output files in the `logs/` directory.

    **To verify the code is working and new log files are being generated:**
    You can first remove the existing `logs/` directory and then re-run the experiment:

    ```bash
    rm -rf logs/
    ./experiment.sh
    ```


### Viewing the Plots

To visualize the experimental results:

1.  Open the `plot.ipynb` Jupyter Notebook.
2.  Run all cells within the notebook to generate and display the plots.