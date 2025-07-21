#!/bin/bash
./exprun.sh
ps aux | grep -E './t1|./t2|./server' | grep -v grep | awk '{print $2}' | xargs kill -9
segments=$(ipcs -m | awk 'NR>3 {print $2}')
echo "$segments"
# Iterate over each segment and detach it forcefully
for segment in $segments; do
    ipcrm -m $segment
done

./t1 0.8 0.600 200 a & ./t2 0.8 0.400 200 a &
wait

# ./t1 0.8 0.800 200 b & ./t2 0.8 0.400 200 b &
# wait


# ./t1 0.8 0.800 200 c & ./t2 0.8 0.400 200 c &
# wait