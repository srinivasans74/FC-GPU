#!/bin/bash
# closedloop.sh

# Function to clean up shared memory segments
cleanup_shared_memory() {
    echo "Cleaning shared memory segments..."
    segments=$(ipcs -m | awk 'NR>3 {print $2}')
    for segment in $segments; do
        ipcrm -m "$segment" 2>/dev/null
    done
}

# Function to clear logs
clear_logs() {
    echo "Clearing log files..."
    cd logs/ || { echo "Error: 'logs/' directory not found."; exit 1; }
    rm -f *
    cd ..
}

# Function to kill running processes for t1, t2, and server
kill_processes() {
    echo "Stopping any running processes..."
    ps aux | grep -E './t1|./t2|./server' | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null
}

# Main execution
cleanup_shared_memory
clear_logs
kill_processes

echo "Compiling CUDA and server programs..."
nvcc -o t1 mm.cu 
nvcc -o t2 stencil.cu -DT2
g++ -o server server.cpp -lrt -pthread -std=c++17

# Define periods and setpoints
periods=(4 15)
setpoints=(0.9 0.9)
duration=200  # Duration in ms or s depending on runit.py

# Launch the experiment
args=(2 t1 t2 "${setpoints[0]}" "${periods[0]}" "${setpoints[1]}" "${periods[1]}" "$duration")

echo "Launching runit.py with arguments: ${args[@]}"
python3 runit.py "${args[@]}"
if [[ $? -ne 0 ]]; then
    echo "Error: runit.py execution failed."
    exit 1
fi

# Plotting
echo "Running pyplot.py for setpoints: ${setpoints[@]}"
python3 pyplot.py "${setpoints[0]}"  "${setpoints[1]}"

echo "Experiment finished successfully."