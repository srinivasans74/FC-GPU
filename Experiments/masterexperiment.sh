#!/bin/bash

# Run Open-loop experiments (Figure 6a, 6b)
cd open-loop/
./experiment.sh
cd ..

# Run Closed-loop experiments (Figure 6c, 6d)
cd closed-loop/
./experiment.sh
cd ..

# Run GPU-TDMH and Closed-loop comparison (Figure 7a–7d)
cd GPU_TDMh_and_closed_loop/
./masterexperiment.sh
cd ..

# Run Ad-hoc controller experiments (custom step sizes)
cd Ad-hoc/
./experiment.sh
cd ..

# Run SISO controller experiments (gain-tuned via MATLAB)
cd SISO/
./experiment.sh
cd ..

# Run FC-GPU integrated controller experiments (Figure 9c, 9d)
cd FC-GPU/
./experiment.sh
cd ..

#onlineexperiments
cd online_new_task_arrival_nvidia/
./experiment.sh
cd ..

#AMD GPU
cd FC-GPU_amd/
./experiment.sh
cd ..