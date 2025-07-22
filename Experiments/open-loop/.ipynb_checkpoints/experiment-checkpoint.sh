#!/bin/bash
# closedloop.sh — Automates real-time CUDA experiment setup

# Function to clean up shared memory segments
cleanup_shared_memory() {
    echo "Cleaning shared memory segments..."
    ipcs -m | awk 'NR>3 {print $2}' | xargs -r -n 1 ipcrm -m 2>/dev/null
}

# Function to clear logs
clear_logs() {
    echo "Clearing log files..."
    if [ -d logs/ ]; then
        rm -f logs/*
    else
        echo "logs/ directory not found. Creating it."
        mkdir -p logs
    fi
}

# Function to kill old task/server processes
kill_processes() {
    echo "Stopping any running processes..."
    pkill -f ./t1 2>/dev/null
    pkill -f ./t2 2>/dev/null
    pkill -f ./server 2>/dev/null
}

# Main script starts here
cleanup_shared_memory
clear_logs
kill_processes

echo "Compiling CUDA and server programs..."
nvcc -o t1 openloopmmt1.cu || { echo "Failed to compile t1"; exit 1; }
nvcc -o t2 openloopstencil.cu || { echo "Failed to compile t2"; exit 1; }

# Experiment configuration
setpoint1=0.9
period1=20
setpoint2=0.9
period2=120
duration=200  # in seconds

echo "Starting real-time CUDA tasks..."
./t1 "$setpoint1" "$period1" "$duration" &
./t2 "$setpoint2" "$period2" "$duration" &
wait 

echo "Running visualization..."
python3 pyplot.py "$setpoint1" "$setpoint2" 

echo "Experiment complete!"
