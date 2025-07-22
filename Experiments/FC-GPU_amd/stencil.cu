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
typedef double Real;
Real *h_Vm, *h_dVm, *h_sigma;   // host views (pinned, mapped)
Real *d_Vm, *d_dVm, *d_sigma;   // GPU aliases returned by cudaHostGetDevicePointer
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

    #define V0(y,z) sm_psi[pii][y][z]
    #define V1(y,z) sm_psi[cii][y][z]
    #define V2(y,z) sm_psi[nii][y][z]

    #define sigmaX(x,y,z,dir) d_sigmaX[ z + nz * ( y + ny * ( x + nx * dir ) ) ]
    #define sigmaY(x,y,z,dir) d_sigmaY[ z + nz * ( y + ny * ( x + nx * dir ) ) ]
    #define sigmaZ(x,y,z,dir) d_sigmaZ[ z + nz * ( y + ny * ( x + nx * dir ) ) ]

    #define psi(x,y,z) d_psi[ z + nz * ( y + ny * ( x ) ) ]
    #define npsi(x,y,z) d_npsi[ z + nz * ( y + ny * ( x ) ) ]

    d_psi = &(psi(XTILE * blockIdx.x, (BSIZE - 2) * blockIdx.y, (BSIZE - 2) * blockIdx.z));
    d_npsi = &(npsi(XTILE * blockIdx.x, (BSIZE - 2) * blockIdx.y, (BSIZE - 2) * blockIdx.z));

    int nLast_x = XTILE + 1;
    int nLast_y = (BSIZE - 1);
    int nLast_z = (BSIZE - 1);

    if (blockIdx.x == gridDim.x - 1) nLast_x = nx - 2 - XTILE * blockIdx.x + 1;
    if (blockIdx.y == gridDim.y - 1) nLast_y = ny - 2 - (BSIZE - 2) * blockIdx.y + 1;
    if (blockIdx.z == gridDim.z - 1) nLast_z = nz - 2 - (BSIZE - 2) * blockIdx.z + 1;

    if (tjj > nLast_y || tkk > nLast_z) return;

    int pii = 0, cii = 1, nii = 2, tii;
    sm_psi[cii][tjj][tkk] = psi(0, tjj, tkk);
    sm_psi[nii][tjj][tkk] = psi(1, tjj, tkk);
    Real xcharge, ycharge, zcharge, dV = 0;

    __syncthreads();

    for (int ii = 1; ii < nLast_x; ii++) {
        sm_psi[nii][tjj][tkk] = psi(ii + 1, tjj, tkk);
        __syncthreads();

        // Compute and accumulate charges here (omitted for brevity)

        __syncthreads();
        tii = pii;
        pii = cii;
        cii = nii;
        nii = tii;
    }
}


void allocateAndInitMemory()
{
    // page-locked host memory that the GPU can directly address
    cudaSetDeviceFlags(cudaDeviceMapHost);        

    cudaHostAlloc(&h_Vm,    sizeof(Real) * vol,     cudaHostAllocMapped);
    cudaHostAlloc(&h_sigma, sizeof(Real) * vol * 9, cudaHostAllocMapped);
    cudaHostAlloc(&h_dVm,   sizeof(Real) * vol,     cudaHostAllocMapped);

    // obtain the matching device pointers (aliases)
    cudaHostGetDevicePointer((void**)&d_Vm,    h_Vm,    0);
    cudaHostGetDevicePointer((void**)&d_sigma, h_sigma, 0);
    cudaHostGetDevicePointer((void**)&d_dVm,   h_dVm,   0);

    // initialise on host – no cudaMemcpy needed
    for (int i = 0; i < vol;      ++i) h_Vm[i]    = i % 19;
    for (int i = 0; i < vol * 9;  ++i) h_sigma[i] = i % 19;
    cudaMemset(h_dVm, 0, sizeof(Real) * vol);  // still fine: h_dVm is host
}

void freeMemory()
{
    cudaFreeHost(h_Vm);
    cudaFreeHost(h_sigma);
    cudaFreeHost(h_dVm);   // releases both host & device views
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
        next_release += std::chrono::milliseconds((long)(sharedData->newperiods[SHARED_MEM_INDEX]));
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
