#!/bin/bash

# Kill any running tasks and clean shared memory segments
echo "Killing running tasks and cleaning shared memory segments..."
ps aux | grep -E './t[0-9]|./server' | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null

segments=$(ipcs -m | awk 'NR>3 {print $2}')
for segment in $segments; do
    ipcrm -m $segment 2>/dev/null
done

# Clear logs
echo "Clearing log files..."
cd logs/ || { echo "Error: 'logs/' directory not found."; exit 1; }
touch as.cpp
rm -f *
cd ..

# List of available tasks
tasks=("dxtc" "stereodisparity" "srad" "pathfinder" "mm" "hotspot" "graph" "floyd" "stencil")
echo "Available tasks:"
for i in "${!tasks[@]}"; do
    echo "$((i + 1)). ${tasks[$i]}"
done

# Ask for the number of tasks
read -p "How many tasks do you want to compile (1-${#tasks[@]})? " num_tasks

# Validate the number of tasks
if ((num_tasks < 1 || num_tasks > ${#tasks[@]})); then
    echo "Error: Please enter a valid number of tasks."
    exit 1
fi



# Ask for the specific task numbers in a single input
read -p "Enter the task numbers separated by spaces (e.g., 2 4 5): " -a task_numbers

# Validate the task numbers
if [[ ${#task_numbers[@]} -ne $num_tasks ]]; then
    echo "Error: You must enter exactly $num_tasks task numbers."
    exit 1
fi

for task_num in "${task_numbers[@]}"; do
    if ((task_num < 1 || task_num > ${#tasks[@]})); then
        echo "Invalid task number: $task_num. Please try again."
        exit 1
    fi
done

# Compile the selected tasks
echo "Compiling the following tasks:"
index=1  # Track object numbering sequentially (e.g., t1, t2, ...)

for task_num in "${task_numbers[@]}"; do
    task_name="${tasks[$((task_num - 1))]}"

    echo "Compiling $task_name as t$index..."

    if [[ $task_name == "dxtc" ]]; then
        nvcc -o "t$index" "${task_name}.cu" -I ../Common/ -I ../Samples/5_Domain_Specific/dxtc/ -L ../Common/ -lcudart -DT$index
    else
        nvcc -o "t$index" "${task_name}.cu" -I ../Common/ -L ../Common/ -lcudart -DT$index
    fi

    ((index++))  # Increment the object number for the next task
done

# Compile the server
echo "Compiling server..."
g++ server.cpp -o server -std=c++11 -lrt -lpthread

echo "Compilation completed."
