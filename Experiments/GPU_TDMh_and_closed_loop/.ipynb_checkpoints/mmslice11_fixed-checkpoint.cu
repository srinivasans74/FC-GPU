#include <cuda_runtime.h>
#include <iostream>
#include <thread>
#include <chrono>
#include <fstream>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <unistd.h>
#include <ctime>       // for clock_nanosleep
#include <future>
#include <functional>
#include "shared_data1.h"

using namespace std;
using namespace std::chrono;

float period, setpoint, termination;
int N = 2048;
const int JOBS = 1;

SharedData* sharedData = nullptr;
bool createdshm = false;
steady_clock::time_point preemptionlaunch;

#ifndef SIGNAL_TYPE
    #ifdef T1
        #define SIGNAL_TYPE SIGUSR1
        #define SHARED_MEM_INDEX 0
        #define LOG_SUFFIX "1"
    #elif defined(T2)
        #define SIGNAL_TYPE SIGUSR2
        #define SHARED_MEM_INDEX 1
        #define LOG_SUFFIX "2"
    #endif
#endif

#define MAKE_LOG_PATH(prefix) ("logs/" prefix LOG_SUFFIX ".txt")
ofstream period1(MAKE_LOG_PATH("p"));
ofstream rtj1(MAKE_LOG_PATH("rtj"));
ofstream preemptiontime(MAKE_LOG_PATH("preemptiont"), ios::app);

float *d_A, *d_B, *d_C;
float *h_A, *h_B, *h_C;

__global__ void matMulKernel(float* A, float* B, float* C, int N, int oy) {
    int row = (blockIdx.y + oy) * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float val = 0;
        for (int k = 0; k < N; ++k)
            val += A[row*N + k] * B[k*N + col];
        C[row*N + col] = val;
    }
}

void kernellaunch(dim3 gridDim, dim3 blockDim, int size, int slices) {
    auto pre = steady_clock::now();
    auto pre_ms = duration_cast<milliseconds>(pre - preemptionlaunch).count();
    preemptiontime << "Preemption=" << pre_ms << " ms\n";

    int sliceY = (gridDim.y + slices - 1) / slices;
    float total_gpu = 0;
    auto cpu_start = high_resolution_clock::now();

    for (int j = 0; j < JOBS; ++j) {
        float job_gpu = 0;
        for (int s = 0; s < slices; ++s) {
            int oy = s * sliceY;
            int blkY = min(sliceY, int(gridDim.y - oy));
            if (blkY <= 0) break;
            dim3 g(gridDim.x, blkY);
            cudaEvent_t st, sp;
            cudaEventCreate(&st); cudaEventCreate(&sp);
            cudaEventRecord(st);
            matMulKernel<<<g, blockDim>>>(d_A, d_B, d_C, N, oy);
            cudaEventRecord(sp); cudaEventSynchronize(sp);
            float el;
            cudaEventElapsedTime(&el, st, sp);
            cudaEventDestroy(st); cudaEventDestroy(sp);
            job_gpu += el;
        }
        total_gpu += job_gpu;
    }

    auto cpu_end = high_resolution_clock::now();
    float cpu_ms = duration<float, milli>(cpu_end - cpu_start).count();
    sharedData->values[SHARED_MEM_INDEX] = pre_ms + cpu_ms / JOBS;
    sharedData->executiontime[SHARED_MEM_INDEX] = total_gpu / JOBS;
}

void runPeriodicTask(
    function<void()> task,
    long period_ms,
    long duration_sec,
    function<void(long)> onPreemption = nullptr
) {
    timespec start_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    timespec next_release = start_time;

    while (true) {
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next_release, nullptr);

        timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        long elapsed_sec = now.tv_sec - start_time.tv_sec;

        if (elapsed_sec >= duration_sec) break;

        long preempt_ms = (now.tv_sec - next_release.tv_sec) * 1000 +
                          (now.tv_nsec - next_release.tv_nsec) / 1000000;
        if (onPreemption && preempt_ms > 0)
            onPreemption(preempt_ms);

        preemptionlaunch = steady_clock::now();
        std::future<void> fut = std::async(std::launch::async, task);
        fut.get();  // blocking wait

        next_release.tv_nsec += period_ms * 1e6;
        while (next_release.tv_nsec >= 1e9) {
            next_release.tv_nsec -= 1e9;
            next_release.tv_sec += 1;
        }
    }
}

void runService(float sp, float per, float term) {
    int size = N * N;

    cudaMalloc(&d_A, size * sizeof(float));
    cudaMalloc(&d_B, size * sizeof(float));
    cudaMalloc(&d_C, size * sizeof(float));
    cudaMallocHost(&h_A, size * sizeof(float));
    cudaMallocHost(&h_B, size * sizeof(float));
    cudaMallocHost(&h_C, size * sizeof(float));

    for (int i = 0; i < size; ++i) h_A[i] = h_B[i] = i % 100;
    cudaMemcpy(d_A, h_A, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16), gridDim((N + 15) / 16, (N + 15) / 16);
    sharedData->newperiods[SHARED_MEM_INDEX] = per;
    int slices = max(1, sharedData->slices[SHARED_MEM_INDEX]);

    auto start_time = steady_clock::now();
    bool check = false;

    runPeriodicTask(
        [&]() {
            auto now = steady_clock::now();
            float elapsed = duration<float>(now - start_time).count();

            if (elapsed >= termination / 2 && !check) {
                check = true;
                cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
                cudaFreeHost(h_A); cudaFreeHost(h_B); cudaFreeHost(h_C);

                N = int(N * 1.2);
                size = N * N;
                cout << "\n[Workload Change] N = " << N << "\n";

                cudaMallocHost(&h_A, size * sizeof(float));
                cudaMallocHost(&h_B, size * sizeof(float));
                cudaMallocHost(&h_C, size * sizeof(float));
                cudaMalloc(&d_A, size * sizeof(float));
                cudaMalloc(&d_B, size * sizeof(float));
                cudaMalloc(&d_C, size * sizeof(float));
                for (int i = 0; i < size; ++i) h_A[i] = h_B[i] = i % 100;
                cudaMemcpy(d_A, h_A, size * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_B, h_B, size * sizeof(float), cudaMemcpyHostToDevice);

                blockDim = dim3(32, 32);
                gridDim = dim3((N + 31) / 32, (N + 31) / 32);
            }

            kernellaunch(gridDim, blockDim, size, slices);
            period1 << sharedData->newperiods[SHARED_MEM_INDEX] << '\n';
            rtj1 << sharedData->values[SHARED_MEM_INDEX] << '\n';
        },
        static_cast<long>(per),
        static_cast<long>(term),
        [&](long preempt_ms) {
            preemptiontime << "Preemption=" << preempt_ms << " ms\n";
        }
    );

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFreeHost(h_A); cudaFreeHost(h_B); cudaFreeHost(h_C);
}

int main(int argc, char* argv[]) {
    if (argc < 4) return 1;
    setpoint = atof(argv[1]);
    period = atof(argv[2]);       // ms
    termination = atof(argv[3]);  // seconds

    key_t key = ftok("shmfile", 65);
    int shmid = shmget(key, 1024, 0666 | IPC_CREAT);
    if (shmid > 0) {
        sharedData = (SharedData*)shmat(shmid, nullptr, 0);
        createdshm = true;
    } else {
        shmid = shmget(key, sizeof(SharedData), 0666);
        sharedData = (SharedData*)shmat(shmid, nullptr, 0);
    }

    std::cout << "[T" << SHARED_MEM_INDEX << "] Soft Real-Time Service running.\n";
    runService(setpoint, period, termination);

    if (createdshm) {
        shmdt(sharedData);
        shmctl(shmid, IPC_RMID, NULL);
    }
    return 0;
}