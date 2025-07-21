#ifndef SHARED_DATA_H
#define SHARED_DATA_H

struct SharedData {
    float values[4]        = {0};   // GPU response-time samples (ms)
    float executiontime[4] = {0};   // CPU response-time samples (ms)
    float newperiods[4]    = {0};   // desired periods (ms)
    int   slices[4]        = {4};   // number of slices per task
};

#endif