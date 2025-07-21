#!/bin/bash

# Function to clean up shared memory segments
cleanup_shared_memory() {
    echo "Cleaning shared memory segments..."
    segments=$(ipcs -m | awk 'NR>3 {print $2}')
    for segment in $segments; do
        ipcrm -m "$segment" 2>/dev/null
    done
}

# Function to kill running tasks and clean logs
cleanup_tasks() {
    echo "Killing running tasks and cleaning shared memory..."
    ps aux | grep -E './t[0-9]|./server' | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null
    cleanup_shared_memory
    echo "Clearing log files..."
    rm -rf logs/* figures/*
}


# Compile server
compile_server() {
    echo "Compiling server..."
    g++ -std=c++17 -O2 -pthread  -lrt -lpthread overhead.cpp -o server 
}



# Main Execution
cleanup_tasks
compile_server
./server
python3 plot.py