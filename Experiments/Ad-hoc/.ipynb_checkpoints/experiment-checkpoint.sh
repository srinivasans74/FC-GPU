#!/bin/bash

# === compilatiolon===
./exprun.sh

# === Kill any leftover processes (t1, t2, server) ===
echo "Killing lingering processes..."
ps aux | grep -E './t1|./t2|./server' | grep -v grep | awk '{print $2}' | xargs -r kill -9

# === Clean up shared memory segments ===
echo "Cleaning shared memory segments..."
segments=$(ipcs -m | awk 'NR>3 {print $2}')
for segment in $segments; do
    ipcrm -m "$segment"
done

# === Define setpoint, period, duration ===
SETPOINT=0.8
PERIOD_T1=0.050
PERIOD_T2=1
DURATION=200

# === Run t1 and t2 for stepsize a ===
echo "Running with stepsize 'a'"
./t1 "$SETPOINT" "$PERIOD_T1" "$DURATION" a & 
./t2 "$SETPOINT" "$PERIOD_T2" "$DURATION" a &
wait

# === Run t1 and t2 for stepsize b ===
echo "Running with stepsize 'b'"
./t1 "$SETPOINT" "$PERIOD_T1" "$DURATION" b & 
./t2 "$SETPOINT" "$PERIOD_T2" "$DURATION" b &
wait

# === Run t1 and t2 for stepsize c ===
echo "Running with stepsize 'c'"
./t1 "$SETPOINT" "$PERIOD_T1" "$DURATION" c & 
./t2 "$SETPOINT" "$PERIOD_T2" "$DURATION" c &
wait

echo "All runs complete."