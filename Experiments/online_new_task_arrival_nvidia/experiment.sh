#!/bin/bash
# exprun.sh - Compile and execute CUDA tasks with server coordination and logging

set -e  # Exit immediately on error

# === Cleanup on Exit or Interrupt ===
trap cleanup_shared_memory EXIT

# --- Functions ---

cleanup_shared_memory() {
    echo "[INFO] Cleaning shared memory segments..."
    ipcs -m | awk 'NR>3 {print $2}' | xargs -r -n 1 ipcrm -m 2>/dev/null || true
}

clear_logs() {
    echo "[INFO] Clearing log files..."
    if [[ -d logs ]]; then
        rm -rf logs/*
    else
        echo "[WARN] 'logs/' directory not found. Creating it..."
        mkdir -p logs
    fi
}

kill_processes() {
    echo "[INFO] Stopping any running tasks or server..."
    pkill -f './t[1-4]' || true
    pkill -f './server' || true
}

compile_tasks() {
    echo "[INFO] Compiling CUDA tasks..."
    local idx=1
    for task_name in "$@"; do
        echo "  [TASK] Compiling ${task_name}.cu as t${idx}..."
        nvcc -o "t${idx}" "${task_name}.cu" -I ../Common/ -L ../Common/ -lcudart -DT${idx} -lpthread
        ((idx++))
    done
}

compile_server() {
    echo "[INFO] Compiling server..."
    g++ server.cpp -o server -std=c++11 -lrt -lpthread -DCOMP
}

handle_logs() {
    local setpoint="$1"
    local comb_name
    comb_name=$(echo "$comb_key" | tr ' ' '_')
    local log_dir="logs/${comb_name}/setpoint_${setpoint// /_}"

    echo "[INFO] Archiving logs to: $log_dir"
    mkdir -p "$log_dir"
    cp logs/* "$log_dir/" 2>/dev/null || true
    mv figures/* "$log_dir/" 2>/dev/null || true
}

# --- Main Execution ---

clear_logs
kill_processes

# Define task combinations
declare -A task_combinations=(
    ["2_memory_2_compute"]="mm srad hotspot stencil"
)

# Run each task combination
for comb_key in "${!task_combinations[@]}"; do
    workloads="${task_combinations[$comb_key]}"
    read -ra workload_list <<< "$workloads"

    echo "[INFO] Processing combination: $comb_key"
    compile_tasks "${workload_list[@]}"
    compile_server

    # Define periods and setpoints
    periods=(0.030 0.030 0.90 0.051)
    setpoints=("0.90")
    duration=400
    for setpoint in "${setpoints[@]}"; do
        interleaved=()
        for i in "${!periods[@]}"; do
            interleaved+=("$setpoint" "${periods[i]}")
        done

        echo "[RUN] python3 runit.py 4 t1 t2 t3 t4 ${interleaved[*]} $duration"
        python3 runit.py 4 t1 t2 t3 t4 "${interleaved[@]}" $duration

        echo "[PLOT] Generating figures for setpoint: $setpoint"
        python3 pyplot.py "$setpoint" "$setpoint" "$setpoint" "$setpoint"

        handle_logs "$setpoint"
        echo "[DONE] Finished setpoint: $setpoint for $comb_key"
    done
done

echo "[COMPLETE] All task combinations executed successfully."