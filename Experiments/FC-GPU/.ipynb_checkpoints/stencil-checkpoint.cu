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

#define BSIZE 16
#define XTILE 20

const int size = 512;
int repeat = 100;
const int nx = size, ny = size, nz = size;
const int vol = nx * ny * nz;

typedef double Real;
Real *d_Vm, *d_dVm, *d_sigma;

steady_clock::time_point start_time, next_release, preemptionlaunch;

__global__ void stencil3d(
    const Real *__restrict__ d_psi,
    Real *__restrict__ d_npsi,
    const Real *__restrict__ d_sigmaX,
    const Real *__restrict__ d_sigmaY,
    const Real *__restrict__ d_sigmaZ,
    int nx, int ny, int nz)
{
    __shared__ Real sm_psi[4][BSIZE][BSIZE];

    const int tjj = threadIdx.y;
    const int tkk = threadIdx.x;

    int pii = 0, cii = 1, nii = 2, tii;

    d_psi = &(d_psi[XTILE * blockIdx.x + nx * ((BSIZE - 2) * blockIdx.y + ny * (BSIZE - 2) * blockIdx.z)]);
    d_npsi = &(d_npsi[XTILE * blockIdx.x + nx * ((BSIZE - 2) * blockIdx.y + ny * (BSIZE - 2) * blockIdx.z)]);

    if (tjj >= BSIZE || tkk >= BSIZE) return;

    sm_psi[cii][tjj][tkk] = d_psi[tjj * nz + tkk];
    sm_psi[nii][tjj][tkk] = d_psi[nx * ny + tjj * nz + tkk];

    __syncthreads();

    for (int ii = 1; ii < XTILE; ++ii) {
        sm_psi[nii][tjj][tkk] = d_psi[(ii + 1) * ny * nz + tjj * nz + tkk];
        __syncthreads();
        __syncthreads();
        tii = pii; pii = cii; cii = nii; nii = tii;
    }
}

void allocateAndInitMemory() {
    cudaMallocHost((void **)&d_Vm, sizeof(Real) * vol);
    cudaMallocHost((void **)&d_sigma, sizeof(Real) * vol * 9);
    cudaMallocHost((void **)&d_dVm, sizeof(Real) * vol);

    for (int i = 0; i < vol; i++) d_Vm[i] = i % 19;
    for (int i = 0; i < vol * 9; i++) d_sigma[i] = i % 19;

    cudaMemset(d_dVm, 0, sizeof(Real) * vol);
}

void freeMemory() {
    cudaFreeHost(d_Vm);
    cudaFreeHost(d_sigma);
    cudaFreeHost(d_dVm);
}

void kernellaunch(dim3 gridDim, dim3 blockDim) {
    auto pre = steady_clock::now();
    float pre_ms = std::chrono::duration<float, std::milli>(pre - preemptionlaunch).count();
    preemptiontime << "Preemption=" << std::fixed << std::setprecision(3) << pre_ms << " ms\n";

    float total_gpu = 0;
    for (int j = 0; j < JOBS; ++j) {
        cudaEvent_t st, sp;
        cudaEventCreate(&st); cudaEventCreate(&sp);
        cudaEventRecord(st);
        for (int r = 0; r < repeat; ++r) {
            stencil3d<<<gridDim, blockDim>>>(d_Vm, d_dVm, d_sigma, d_sigma + 3 * vol, d_sigma + 6 * vol, nx, ny, nz);
        }
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
    allocateAndInitMemory();
    int bdimx = (nx - 2) / XTILE + ((nx - 2) % XTILE != 0);
    int bdimy = (ny - 2) / (BSIZE - 2) + ((ny - 2) % (BSIZE - 2) != 0);
    int bdimz = (nz - 2) / (BSIZE - 2) + ((nz - 2) % (BSIZE - 2) != 0);

    dim3 gridDim(bdimx, bdimy, bdimz);
    dim3 blockDim(BSIZE, BSIZE, 1);
    sharedData->newperiods[SHARED_MEM_INDEX] = period_ms;

    start_time = steady_clock::now();
    next_release = start_time;
    const float change_trigger = termination_ms / 2.0f;

    while (true) {
        std::this_thread::sleep_until(next_release);
        auto now = steady_clock::now();
        float elapsed_ms = std::chrono::duration<float, std::milli>(now - start_time).count();
        if (elapsed_ms >= termination_ms) break;

   

        preemptionlaunch = steady_clock::now();
        future<void> fut = async(launch::async, [&]() {
            kernellaunch(gridDim, blockDim);
        });
        fut.get();

        period1 << period_ms << '\n';
        rtj1 << sharedData->values[SHARED_MEM_INDEX] << '\n';
        next_release += std::chrono::milliseconds((long)(period_ms));
    }

    freeMemory();
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        cerr << "Usage: ./stencil <setpoint> <period(ms)> <termination(s)>\n";
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

    cout << "[T" << SHARED_MEM_INDEX << "] Real-Time Stencil Task Started.\n";
    runPeriodicService();

    if (createdshm) {
        shmdt(sharedData);
        shmctl(shmid, IPC_RMID, NULL);
    }

    outputfile.close(); rtj1.close(); period1.close(); preemptiontime.close();
    return 0;
}
