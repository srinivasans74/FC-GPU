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
#include "shared_data.h"

float period;
float setpoint;
float termination;
using namespace std;
int pids;
typedef double Real;
char stepsize;

#ifndef SIGNAL_TYPE
    #ifdef T1
        #define SIGNAL_TYPE SIGUSR1  // For T1: SIGUSR1
        #define SHARED_MEM_INDEX 0   // Index: 0
        #define SHARED_MEM_INDEX2 0  // Index2: 0
        #define LOG_SUFFIX "1"  // Default suffix if no flag is provided
    #elif defined(T2)
        #define SIGNAL_TYPE SIGUSR2  // For T2: SIGUSR2
        #define SHARED_MEM_INDEX 1   // Index: 1
        #define SHARED_MEM_INDEX2 1  // Index2: 1
        #define LOG_SUFFIX "2"  // Default suffix if no flag is provided
    #elif defined(T3)
        #define SIGNAL_TYPE SIGUSR3  // For T3: SIGUSR3
        #define SHARED_MEM_INDEX 4   // Index: 4
        #define SHARED_MEM_INDEX2 2  // Index2: 2
        #define LOG_SUFFIX "3"  // Default suffix if no flag is provided
    #elif defined(T4)
        #define SIGNAL_TYPE SIGUSR4  // For T4: SIGUSR4
        #define SHARED_MEM_INDEX 5   // Index: 5
        #define SHARED_MEM_INDEX2 3  // Index2: 3
        #define LOG_SUFFIX "4"  // Default suffix if no flag is provided
    #else
        #define SIGNAL_TYPE SIGUSR1  // Default: SIGUSR1
        #define SHARED_MEM_INDEX 0   // Default Index: 0
        #define SHARED_MEM_INDEX2 0  // Default Index2: 0
        #define LOG_SUFFIX "1"  // Default suffix if no flag is provided
    #endif
#endif

// Macro to build base log file names
#define MAKE_LOG_PATH(prefix) ("logs/" prefix LOG_SUFFIX ".txt")

// Global log file paths (will be reinitialized in main to include the stepsize letter)
std::string outputFilePath = MAKE_LOG_PATH("log");
std::string slackLogPath = MAKE_LOG_PATH("s");
std::string periodLogPath = MAKE_LOG_PATH("p");
std::string rtjLogPath = MAKE_LOG_PATH("rtj");
std::string preemptionLogPath = MAKE_LOG_PATH("preemptiont");
std::string rtrdeadlinePath = MAKE_LOG_PATH("rtrdeadline");
std::string executionfilepath  = "logs/et2.txt";

std::ofstream outputfile(outputFilePath);
std::ofstream slack1(slackLogPath);
std::ofstream period1(periodLogPath);
std::ofstream rtj1(rtjLogPath);
std::ofstream preemptiontime(preemptionLogPath, std::ios::app);
std::ofstream rtrdeadline(rtrdeadlinePath);
std::ofstream executiontimef(executionfilepath);
std::vector<float> executiontime;
#define BSIZE 16
#define XTILE 20

const int size = 512;
int repeat = 375;
const int nx = size, ny = size, nz = size;
const int vol = nx * ny * nz;
Real *h_Vm, *h_dVm, *h_sigma;        // host, mapped
Real *d_Vm, *d_dVm, *d_sigma;        // device aliases returned by cudaHostGetDevicePointer
typedef double Real;

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

void init()
{
    FILE *file = fopen("logs/mainpid.txt", "r");
    int value;
    fscanf(file, "%d", &value);
    pids = value;
    fclose(file);
    std::cout << "PID PARENT T1= " << pids << endl;
}
int jobs = 2;
bool keepRunning = true;
bool signalReceived = false;
void handleSignal(int signal) {
    if (signal == SIGHUP) {
        bool signalReceived = true;
    }
}
  
std::vector<float> responsetime;
float min_rate;
float max_rate;

float calculateaverage(std::vector<float> resp_time)
{
    float avg_resp_time = 0;
    for (const auto& time : resp_time) {
        avg_resp_time += time;
    }
    avg_resp_time /= resp_time.size();
    return avg_resp_time;
}
std::chrono::time_point<std::chrono::steady_clock> controltime;
std::chrono::time_point<std::chrono::steady_clock> program_start_time;
std::chrono::time_point<std::chrono::steady_clock> preemptionlaunch;

cudaEvent_t start, stop;

// Initialize CUDA events for timing
void initializeCudaEvents() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
}

// Corrected kernellaunch() receives h_Vm and h_sigma as parameters.
// Reduced redundant transfers by copying h_Vm and h_sigma into GPU pinned memory once.
void kernellaunch(dim3 gridDim, dim3 blockDim, Real* h_Vm, Real* h_sigma) {
    auto start_chrono = std::chrono::steady_clock::now();
    auto preemption = std::chrono::steady_clock::now();
    auto preemption_start_duration = std::chrono::duration_cast<std::chrono::milliseconds>(preemption - preemptionlaunch).count();
    preemptiontime << "Preemption time = " << preemption_start_duration/(1) << " ms" << std::endl;
    

   
    float kernel_elapsed_time = 0.0f;
    float respsum = 0.0f;
    auto cpu_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < jobs; ++i) {
        cudaEvent_t startEvent, stopEvent;
        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);

        // Record the start event
        cudaEventRecord(startEvent);

        // Launch the kernel multiple times based on the repeat count
        for (int j = 0; j < repeat; ++j) {
            stencil3d<<<gridDim, blockDim>>>(d_Vm, d_dVm, d_sigma, d_sigma + 3 * vol, d_sigma + 6 * vol, nx, ny, nz);
        }

        cudaDeviceSynchronize();

        // Record the stop event
        cudaEventRecord(stopEvent);
        cudaEventSynchronize(stopEvent);

        cudaEventElapsedTime(&kernel_elapsed_time, startEvent, stopEvent);
        respsum += kernel_elapsed_time;

        outputfile << "Job time = \t" << kernel_elapsed_time << " ms" << std::endl;

        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
    }
    
    auto cpu_end = std::chrono::high_resolution_clock::now();
    float cpu_time = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();
    responsetime.push_back(respsum/jobs + preemption_start_duration/(1*1));
    executiontime.push_back(cpu_time/jobs);
    outputfile << "responsetime= \t" << respsum/jobs + preemption_start_duration/(1) << " ms" << std::endl;

}

float calculatePercentage(const vector<float>& responsetime, float x) {
    int count = 0;
    int total = responsetime.size();
    
    for (float value : responsetime) {
        if (value / x > 1) {
            count++;
        }
    }
    float percentage = static_cast<float>(count) / total * 100;
    return percentage;
}

//
// runService: Uses host memory allocated with malloc and GPU pinned memory (cudaMallocHost)
// Also reinitializes local log file names to include the stepsize letter (from the fourth parameter).
//
void runService(float setpoint, float period, float termination) {
    auto program_start_time = std::chrono::steady_clock::now();
    auto controltime = std::chrono::steady_clock::now();

    printf("Grid dimension: nx=%d ny=%d nz=%d\n", nx, ny, nz);
    cudaSetDeviceFlags(cudaDeviceMapHost);           


    
    cudaHostAlloc((void **)&h_Vm,    sizeof(Real) * vol,     cudaHostAllocMapped);
    cudaHostAlloc((void **)&h_sigma, sizeof(Real) * vol * 9, cudaHostAllocMapped);
    cudaHostAlloc((void **)&h_dVm,   sizeof(Real) * vol,     cudaHostAllocMapped);
    cudaMemset(h_dVm, 0, sizeof(Real) * vol); 

    // GPU aliases that refer to the same physical pages
    cudaHostGetDevicePointer((void **)&d_Vm,    h_Vm,    0);
    cudaHostGetDevicePointer((void **)&d_sigma, h_sigma, 0);
    cudaHostGetDevicePointer((void **)&d_dVm,   h_dVm,   0);
 

    for (int i = 0; i < vol; ++i) h_Vm[i] = i % 19;
    for (int i = 0; i < vol * 9; ++i) h_sigma[i] = i % 19;

    int bdimx = (nx - 2) / XTILE + (((nx - 2) % XTILE) == 0 ? 0 : 1);
    int bdimy = (ny - 2) / (BSIZE - 2) + (((ny - 2) % (BSIZE - 2)) == 0 ? 0 : 1);
    int bdimz = (nz - 2) / (BSIZE - 2) + (((nz - 2) % (BSIZE - 2)) == 0 ? 0 : 1);

    dim3 grids(bdimx, bdimy, bdimz);
    dim3 blocks(BSIZE, BSIZE, 1);
     //Copy host data to GPU pinned memory once (reducing redundant transfers)

    std::future<void> asyncTask;
    std::vector<float> controlperiod;

    initializeCudaEvents(); 

    // Reinitialize local log file names to include the stepsize letter
    std::string s_slack  = "logs/s2"  + std::string(1, stepsize) + ".txt";
    std::string s_period = "logs/p2"  + std::string(1, stepsize) + ".txt";
    std::string s_rtj    = "logs/rtj2" + std::string(1, stepsize) + ".txt";
    std::ofstream slack1(s_slack);
    std::ofstream period1(s_period);
    std::ofstream rtj1(s_rtj);

    bool switched = false;

    while (keepRunning) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(now - program_start_time).count() / 1000.0;
        outputfile << "Task launched at: " << elapsed_time << " sec" << std::endl << std::endl;
        std::chrono::duration<double> controlperiodaa = now - controltime;
  
        if (elapsed_time > termination) {
            std::cout << "Termination time reached. Exiting." << std::endl;
            break;
        }

     
     
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
            float step = (stepsize == 'a') ? 0.145 : (stepsize == 'b') ? 0.045 : 0.095;
            period += (rt / (period * 1000) < setpoint) ? -step : step;
            controltime = chrono::steady_clock::now();
        }

        preemptionlaunch = std::chrono::steady_clock::now();

        if (asyncTask.valid()) {
            asyncTask.get();
        }

        asyncTask = std::async(std::launch::async, [&]() {
            kernellaunch(grids, blocks, h_Vm, h_sigma);
        });

        period1 << period << std::endl;

        std::chrono::milliseconds sleep_duration(static_cast<long long>(period * 1000.0));
        std::this_thread::sleep_for(sleep_duration);
    }

    if (asyncTask.valid()) {
        asyncTask.get();
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFreeHost(h_Vm);
    cudaFreeHost(h_sigma);
    cudaFreeHost(h_dVm);      // releases host+device views in one call

    slack1.close();
    period1.close();
    rtj1.close();
}

int main(int argc, char *argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <setpoint> <period> <termination> <stepsize (a, b, or c)>" << std::endl;
        return 1;
    }

    setpoint = atof(argv[1]);
    period = atof(argv[2]);
    termination = atof(argv[3]);
    stepsize = argv[4][0];
    min_rate = period - period * 0.90;
    max_rate = period + period * 4.90;
    std::signal(SIGHUP, handleSignal);

    // Reinitialize global log file names to include stepsize letter
    std::string new_outputFilePath  = "logs/log2" + std::string(2, stepsize) + ".txt";
    std::string new_rtrdeadlinePath = "logs/rtrdeadlinet2" + std::string(1, stepsize) + ".txt";
    preemptionLogPath= "logs/preemptiont" + std::string(LOG_SUFFIX) + std::string(1, stepsize) + ".txt";
    rtrdeadlinePath  = "logs/rtrdeadline" + std::string(LOG_SUFFIX) + std::string(1, stepsize) + ".txt";
    outputfile.close();
    rtrdeadline.close();
    outputfile.open(new_outputFilePath);
    rtrdeadline.open(new_rtrdeadlinePath);

    std::cout << "Min rate = \t" << min_rate << "Max rate = " << max_rate << std::endl;
    std::cout << "Termination = " << termination << std::endl;
    std::cout << "Service is running. Press Ctrl+C to exit." << std::endl;
    runService(setpoint, period, termination);
    outputfile.close();
    std::cout << "Service is shutting down." << std::endl;
    
    return 0;
}