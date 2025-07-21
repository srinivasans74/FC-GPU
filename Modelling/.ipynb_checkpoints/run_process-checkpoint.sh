#!/bin/bash

WIDTH=1600
PERIOD=1
DURATION=10

LOGFILE="response_times.log"
echo -n "" > $LOGFILE

echo "Running with 1 task" >> $LOGFILE
./t1 $WIDTH $PERIOD $DURATION >> $LOGFILE
wait

echo "Running with 2 tasks" >> $LOGFILE
./t1 $WIDTH $PERIOD $DURATION >> $LOGFILE &
./t1 $WIDTH $PERIOD $DURATION >> $LOGFILE &
wait

echo "Running with 3 tasks" >> $LOGFILE
./t1 $WIDTH $PERIOD $DURATION >> $LOGFILE &
./t1 $WIDTH $PERIOD $DURATION >> $LOGFILE &
./t1 $WIDTH $PERIOD $DURATION >> $LOGFILE &
wait

echo "Running with 4 tasks" >> $LOGFILE
./t1 $WIDTH $PERIOD $DURATION >> $LOGFILE &
./t1 $WIDTH $PERIOD $DURATION >> $LOGFILE &
./t1 $WIDTH $PERIOD $DURATION >> $LOGFILE &
./t1 $WIDTH $PERIOD $DURATION >> $LOGFILE &
wait

echo "All runs complete. Output saved to $LOGFILE"
python3 plot.py