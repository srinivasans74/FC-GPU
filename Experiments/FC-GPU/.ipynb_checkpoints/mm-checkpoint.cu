#include <cuda_runtime.h>
#include <iostream>
#include <csignal>
#include <thread>
#include <future>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <unistd.h>
#include "shared_data.h"

using namespace std;
using steady_clock = std::chrono::steady_clock;

float setpoint, period_ms, termination_ms;
int N = 512;
const int JOBS = 1;
bool workload_changed = false;
bool createdshm = false;

#ifndef SIGNAL_TYPE
    #ifdef T1
        #define SIGNAL_TYPE SIGUSR1
        #define SHARED_MEM_INDEX 0
        #define SHARED_MEM_INDEX2 0
        #define LOG_SUFFIX "1"
    #elif defined(T2)
        #define SIGNAL_TYPE SIGUSR2
        #define SHARED_MEM_INDEX 1
        #define SHARED_MEM_INDEX2 1
        #define LOG_SUFFIX "2"
    #else
        #define SIGNAL_TYPE SIGUSR1
        #define SHARED_MEM_INDEX 0
        #define SHARED_MEM_INDEX2 0
        #define LOG_SUFFIX "1"
    #endif
#endif

#define MAKE_LOG_PATH(prefix) ("logs/" prefix LOG_SUFFIX ".txt")

ofstream outputfile(MAKE_LOG_PATH("log"));
ofstream rtj1(MAKE_LOG_PATH("rtj"));
ofstream period1(MAKE_LOG_PATH("p"));
ofstream preemptiontime(MAKE_LOG_PATH("preemptiont"), ios::app);

float *h_A, *h_B, *h_C;        // host views (pinned, mapped)
float *d_A, *d_B, *d_C;        // device aliases from cudaHostGetDevicePointer
steady_clock::time_point start_time, next_release, preemptionlaunch;

__global__ void matMulKernel(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float val = 0.0f;
        for (int k = 0; k < N; ++k)
            val += A[row * N + k] * B[k * N + col];
        C[row * N + col] = val;
    }
}

void allocateAndInitMemory(int size)
{
    // Page-locked host buffers that the GPU can directly access
    cudaSetDeviceFlags(cudaDeviceMapHost);        
    cudaHostAlloc(&h_A, size * sizeof(float), cudaHostAllocMapped);
    cudaHostAlloc(&h_B, size * sizeof(float), cudaHostAllocMapped);
    cudaHostAlloc(&h_C, size * sizeof(float), cudaHostAllocMapped);

    // Obtain the matching device pointers (aliases, no extra memory)
    cudaHostGetDevicePointer((void**)&d_A, h_A, 0);
    cudaHostGetDevicePointer((void**)&d_B, h_B, 0);
    cudaHostGetDevicePointer((void**)&d_C, h_C, 0);

    // Initialise on host – NO cudaMemcpy needed
    for (int i = 0; i < size; ++i) h_A[i] = h_B[i] = i % 100;
    // h_C will be overwritten by the kernel
}
void freeMemory()
{
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);          // releases both host & device views in one call
}

void kernellaunch(dim3 gridDim, dim3 blockDim, int size) {
    auto pre = steady_clock::now();
    float pre_ms = std::chrono::duration<float, std::milli>(pre - preemptionlaunch).count();
    preemptiontime << "Preemption=" << std::fixed << std::setprecision(3) << pre_ms << " ms\n";

    float total_gpu = 0;
    for (int j = 0; j < JOBS; ++j) {
        cudaEvent_t st, sp;
        cudaEventCreate(&st); cudaEventCreate(&sp);
        cudaEventRecord(st);
        matMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
        cudaEventRecord(sp); cudaEventSynchronize(sp);
        float el;
        cudaEventElapsedTime(&el, st, sp);
        total_gpu += el;
        cudaEventDestroy(st); cudaEventDestroy(sp);
    }

    sharedData->values[SHARED_MEM_INDEX] = pre_ms + total_gpu / JOBS;
    sharedData->executiontime[SHARED_MEM_INDEX] = total_gpu / JOBS;
}

void runPeriodicService() {
    int size = N * N;
    allocateAndInitMemory(size);
    dim3 blockDim(16, 16), gridDim((N + 15) / 16, (N + 15) / 16);
    sharedData->newperiods[SHARED_MEM_INDEX] = period_ms;

    start_time = steady_clock::now();
    next_release = start_time;

    const float change_trigger = termination_ms / 2.0f;
    while (true) {
        std::this_thread::sleep_until(next_release);
        //period_ms
        auto now = steady_clock::now();
        float elapsed_ms = std::chrono::duration<float, std::milli>(now - start_time).count();
        if (elapsed_ms >= termination_ms) break;

    
        preemptionlaunch = steady_clock::now();
        future<void> fut = async(launch::async, [&]() {
            kernellaunch(gridDim, blockDim, size);
        });
        fut.get();

        period1 << period_ms << '\n';
        rtj1 << sharedData->values[SHARED_MEM_INDEX] << '\n';

        next_release += std::chrono::milliseconds((long)(sharedData->newperiods[SHARED_MEM_INDEX]));
    }

    freeMemory();
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        cerr << "Usage: ./program <setpoint> <period(ms)> <termination(ms)>\n";
        return 1;
    }

    setpoint = atof(argv[1]);
    period_ms = atof(argv[2]);
    termination_ms = atof(argv[3]) * 1000.0f;

    key_t key = ftok("shmfile", 65);
    int shmid = shmget(key, 1024, 0666 | IPC_CREAT);
    if (shmid > 0) {
        sharedData = (SharedData*)shmat(shmid, nullptr, 0);
        createdshm = true;
    } else {
        shmid = shmget(key, sizeof(SharedData), 0666);
        sharedData = (SharedData*)shmat(shmid, nullptr, 0);
    }

    cout << "[T" << SHARED_MEM_INDEX << "] Real-Time GPU Task Started.\n";
    runPeriodicService();

    if (createdshm) {
        shmdt(sharedData);
        shmctl(shmid, IPC_RMID, NULL);
    }

    outputfile.close(); rtj1.close(); period1.close(); preemptiontime.close();
    return 0;
}