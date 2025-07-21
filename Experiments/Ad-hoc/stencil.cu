#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <csignal>
#include <thread>
#include <chrono>
#include <future>
#include <vector>
#include <fstream>

float period;
float setpoint;
float termination;
int pids;

using namespace std;
typedef double Real;

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
    #elif defined(T3)
        #define SIGNAL_TYPE SIGUSR3
        #define SHARED_MEM_INDEX 4
        #define SHARED_MEM_INDEX2 2
        #define LOG_SUFFIX "3"
    #elif defined(T4)
        #define SIGNAL_TYPE SIGUSR4
        #define SHARED_MEM_INDEX 5
        #define SHARED_MEM_INDEX2 3
        #define LOG_SUFFIX "4"
    #else
        #define SIGNAL_TYPE SIGUSR1
        #define SHARED_MEM_INDEX 0
        #define SHARED_MEM_INDEX2 0
        #define LOG_SUFFIX "1"
    #endif
#endif

#define MAKE_LOG_PATH(prefix) ("logs/" prefix LOG_SUFFIX)

string outputFilePath;
string slackLogPath;
string periodLogPath;
string rtjLogPath;
string preemptionLogPath;
string rtrdeadlinePath;
string executionfilepath;

ofstream outputfile;
ofstream slack1;
ofstream period1;
ofstream rtj1;
ofstream preemptiontime;
ofstream rtrdeadline;
ofstream executiontimef;

vector<float> responsetime;
vector<float> executiontime;

#define BSIZE 16
#define XTILE 20

const int size = 512;
int repeat = 375;
const int nx = size, ny = size, nz = size;
const int vol = nx * ny * nz;
Real *d_Vm, *d_dVm, *d_sigma;

char stepsize;
bool keepRunning = true;
bool signalReceived = false;
cudaEvent_t start, stop;

typedef double Real;

void handleSignal(int signal) {
    if (signal == SIGHUP) {
        signalReceived = true;
    }
}

float calculateaverage(const vector<float>& times) {
    if (times.empty()) return 0.0f;
    float sum = 0;
    for (const auto& t : times) sum += t;
    return sum / times.size();
}

float calculatePercentage(const vector<float>& times, float deadline_ms) {
    int count = 0;
    for (float t : times) {
        if (t > deadline_ms) count++;
    }
    return times.empty() ? 0.0f : (float)count / times.size() * 100;
}

__global__ void stencil3d(
    const Real *__restrict__ d_psi,
    Real *__restrict__ d_npsi,
    const Real *__restrict__ d_sigmaX,
    const Real *__restrict__ d_sigmaY,
    const Real *__restrict__ d_sigmaZ,
    int nx, int ny, int nz) {

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
        // Compute and accumulate charges here (omitted)
        __syncthreads();
        tii = pii;
        pii = cii;
        cii = nii;
        nii = tii;
    }
}

void initializeCudaEvents() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
}


void kernellaunch(dim3 gridDim, dim3 blockDim, Real* h_Vm, Real* h_sigma) {
    auto preemption = chrono::steady_clock::now();
    auto preemption_start_duration = chrono::duration_cast<chrono::milliseconds>(preemption.time_since_epoch()).count();
    preemptiontime << "Preemption time = " << preemption_start_duration << " ms" << endl;

    float respsum = 0.0f;
    auto cpu_start = chrono::high_resolution_clock::now();

    for (int i = 0; i < 1; ++i) {
        cudaEventRecord(start);

        for (int j = 0; j < repeat; ++j) {
            stencil3d<<<gridDim, blockDim>>>(d_Vm, d_dVm, d_sigma, d_sigma + 3 * vol, d_sigma + 6 * vol, nx, ny, nz);
        }

        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float kernel_time = 0;
        cudaEventElapsedTime(&kernel_time, start, stop);
        respsum += kernel_time;
        outputfile << "Job time = \t" << kernel_time << " ms" << endl;
    }

    auto cpu_end = chrono::high_resolution_clock::now();
    float cpu_time = chrono::duration<float, milli>(cpu_end - cpu_start).count();

    responsetime.push_back(respsum / 2);
    executiontime.push_back(cpu_time / 2);
    outputfile << "responsetime= \t" << respsum / 2 << " ms" << endl;
}

void runService(float setpoint, float period, float termination) {
    Real* h_Vm = (Real*)malloc(sizeof(Real) * vol);
    Real* h_sigma = (Real*)malloc(sizeof(Real) * vol * 9);

    cudaMallocHost((void**)&d_Vm, sizeof(Real) * vol);
    cudaMallocHost((void**)&d_dVm, sizeof(Real) * vol);
    cudaMallocHost((void**)&d_sigma, sizeof(Real) * vol * 9);

    for (int i = 0; i < vol; ++i) h_Vm[i] = i % 19;
    for (int i = 0; i < vol * 9; ++i) h_sigma[i] = i % 19;

    cudaMemcpy(d_Vm, h_Vm, sizeof(Real) * vol, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sigma, h_sigma, sizeof(Real) * vol * 9, cudaMemcpyHostToDevice);
    cudaMemset(d_dVm, 0, sizeof(Real) * vol);

    dim3 grid((nx * ny * nz + 255) / 256);
    dim3 block(256);

    initializeCudaEvents();
    auto start_time = chrono::steady_clock::now();
    auto controltime = start_time;

    future<void> asyncTask;
    while (keepRunning) {
        auto now = chrono::steady_clock::now();
        auto elapsed = chrono::duration_cast<chrono::seconds>(now - start_time).count();

        if (elapsed > termination) break;

        if (chrono::duration<double>(now - controltime).count() >= 4.0) {
            float rt = calculateaverage(responsetime);
            float et = calculateaverage(executiontime);
            float rtr = calculatePercentage(responsetime, period * 1000);
            rtrdeadline << rtr << endl;
            executiontimef << et << endl;
            responsetime.clear();
            executiontime.clear();
            period1 << period << endl;
            rtj1 << rt << endl;
            slack1 << (rt / (period * 1000)) << endl;
            //extreme stepsizes to see different impact
            float step = (stepsize == 'a') ? 0.145 : (stepsize == 'b') ? 0.200 : 0.50;
            period += (rt / (period * 1000) < setpoint) ? -step : step;
            controltime = chrono::steady_clock::now();
        }

        if (asyncTask.valid()) asyncTask.get();
        asyncTask = async(launch::async, [&]() {
            kernellaunch(grid, block, h_Vm, h_sigma);
        });

        this_thread::sleep_for(chrono::milliseconds(static_cast<long long>(period * 1000.0)));
    }

    if (asyncTask.valid()) asyncTask.get();

    cudaFreeHost(d_Vm);
    cudaFreeHost(d_dVm);
    cudaFreeHost(d_sigma);
    free(h_Vm);
    free(h_sigma);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    slack1.close();
    period1.close();
    rtj1.close();
    outputfile.close();
    rtrdeadline.close();
    executiontimef.close();
    preemptiontime.close();
}

int main(int argc, char *argv[]) {
    if (argc < 5) {
        cerr << "Usage: " << argv[0] << " <setpoint> <period> <termination> <stepsize>\n";
        return 1;
    }

    setpoint = atof(argv[1]);
    period = atof(argv[2]);
    termination = atof(argv[3]);
    stepsize = argv[4][0];

    signal(SIGHUP, handleSignal);

    outputFilePath       = "logs/log" LOG_SUFFIX + string(2, stepsize) + ".txt";
    slackLogPath         = "logs/s"   LOG_SUFFIX + string(2, stepsize) + ".txt";
    periodLogPath        = "logs/p"   LOG_SUFFIX + string(2, stepsize) + ".txt";
    rtjLogPath           = "logs/rtj" LOG_SUFFIX + string(2, stepsize) + ".txt";
    preemptionLogPath    = "logs/preemptiont" LOG_SUFFIX + string(2, stepsize) + ".txt";
    rtrdeadlinePath      = "logs/rtrdeadline" LOG_SUFFIX + string(2, stepsize) + ".txt";
    executionfilepath    = "logs/et" LOG_SUFFIX + string(2, stepsize) + ".txt";

    outputfile.open(outputFilePath);
    slack1.open(slackLogPath);
    period1.open(periodLogPath);
    rtj1.open(rtjLogPath);
    preemptiontime.open(preemptionLogPath, ios::app);
    rtrdeadline.open(rtrdeadlinePath);
    executiontimef.open(executionfilepath);

    cout << "Service running. Press Ctrl+C to stop.\n";
    runService(setpoint, period, termination);
    cout << "Service stopped.\n";
    return 0;
}