#!/bin/bash

# -------- Configuration --------
SETPOINT=0.9
PERIOD=20
TERMINATION=200

T1_SRC="mmslice11_fixed.cu"
T2_SRC="mmslice22_fixed.cu"
SERVER_SRC="tdhmaserver_fixed.cpp"

# -------- Functions --------

cleanup_shared_memory() {
    echo "[*] Cleaning shared memory segments..."
    for id in $(ipcs -m | awk 'NR>3 {print $2}'); do
        ipcrm -m "$id" 2>/dev/null
    done
}

kill_processes() {
    echo "[*] Killing existing processes..."
    pkill -f "./t1"
    pkill -f "./t2"
    pkill -f "./server"
}

compile_all() {
    echo "[*] Compiling CUDA tasks..."
    nvcc -std=c++11 -Xcompiler -fPIC -o t1 "$T1_SRC" -DT1 -lpthread
    nvcc -std=c++11 -Xcompiler -fPIC -o t2 "$T2_SRC" -DT2 -lpthread
    echo "    nvcc -std=c++11 -Xcompiler -fPIC -o t1 \"$T1_SRC\" -DT1 -lpthread"
     echo "    nvcc -std=c++11 -Xcompiler -fPIC -o t2 \"$T2_SRC\" -DT2 -lpthread"
    echo "[*] Compiling server..."
    g++ "$SERVER_SRC" -o server -std=c++11 -lrt -lpthread
}

prepare_logs() {
    echo "[*] Preparing log directories..."
    mkdir -p logs tdmalogs tdmalogs1 figures
    rm -rf logs/* tdmalogs/* figures/*
}

run_experiment() {
    echo "[*] Running experiment..."
    ./t1 $SETPOINT $PERIOD $TERMINATION & 
    ./t2 $SETPOINT $PERIOD $TERMINATION & 
    ./server $TERMINATION $SETPOINT
}

finalize_logs() {
    echo "[*] Archiving results..."
    mkdir -p tdmalogs1/
    cp -rp tdmalogs/* tdmalogs1/.
    rm -rf tdmalogs/*
}

# -------- Main --------

cleanup_shared_memory
kill_processes
compile_all
prepare_logs
run_experiment
wait
finalize_logs
python3 gputdhmhplot.py 
echo "[*] Experiment complete!"