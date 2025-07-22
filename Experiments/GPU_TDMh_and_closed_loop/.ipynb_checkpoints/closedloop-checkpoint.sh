#!/bin/bash

# -------- Configuration --------
TASKS=("mmnew1.cu" "mmnew2.cu")
SETPOINTS=(0.90 0.90)
PERIODS=(20 20)  #period is given in seconds here
TERMINATION=200
SERVER_SRC="server.cpp"

# -------- Functions --------

cleanup_shared_memory() {
    echo "[*] Cleaning shared memory segments..."
    for id in $(ipcs -m | awk 'NR>3 {print $2}'); do
        ipcrm -m "$id" 2>/dev/null
    done
}

kill_processes() {
    echo "[*] Killing existing processes..."
    pkill -f "./t[1-9]" 2>/dev/null
    pkill -f "./server" 2>/dev/null
}

compile_tasks() {
    echo "[*] Compiling CUDA tasks..."
    for i in "${!TASKS[@]}"; do
        index=$((i + 1))
        nvcc -o "t$index" "${TASKS[i]}" -I ../Common/ -L ../Common/ -lcudart -DT$index
        echo "    Compiled ${TASKS[i]} -> t$index"
    done
}

compile_server() {
    echo "[*] Compiling server..."
    g++ "$SERVER_SRC" -o server -std=c++11 -lrt -lpthread
}

prepare_logs() {
    echo "[*] Preparing log directories..."
    rm -rf logs/* figures/*
    mkdir -p logs figures
}

run_experiment() {
    echo "[*] Running experiment with setpoints: ${SETPOINTS[*]} and periods: ${PERIODS[*]}"
    
    INTERLEAVED=()
    for i in "${!SETPOINTS[@]}"; do
        INTERLEAVED+=("${SETPOINTS[i]}" "${PERIODS[i]}")
    done

    python3 runit.py "${#TASKS[@]}" $(for i in $(seq 1 ${#TASKS[@]}); do echo -n "t$i "; done) "${INTERLEAVED[@]}" "$TERMINATION"

    if [[ $? -ne 0 ]]; then
        echo "[!] Error: runit.py failed"
        exit 1
    fi

    python3 pyplot.py "${SETPOINTS[@]}"
}

archive_logs() {
    LOG_DIR="logs/setpoint_$(echo "${SETPOINTS[*]}" | tr ' ' '_')"
    mkdir -p "$LOG_DIR"
    cp logs/*.txt "$LOG_DIR/" 2>/dev/null
    mv figures/* "$LOG_DIR/" 2>/dev/null
    echo "[*] Logs archived to $LOG_DIR"
}

# -------- Main --------

cleanup_shared_memory
kill_processes
compile_tasks
compile_server
prepare_logs
run_experiment
archive_logs

echo "[*] All done."