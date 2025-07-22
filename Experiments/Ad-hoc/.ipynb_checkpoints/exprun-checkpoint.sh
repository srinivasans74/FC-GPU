#!/bin/bash
ps aux | grep -E './t1|./t2|./server' | grep -v grep | awk '{print $2}' | xargs kill -9
segments=$(ipcs -m | awk 'NR>3 {print $2}')
echo "$segments"
# Iterate over each segment and detach it forcefully
for segment in $segments; do
    ipcrm -m $segment
done
m=1 

nvcc -o t1 mmt11.cu -DT1
nvcc -o t2 stencil1.cu -DT2

