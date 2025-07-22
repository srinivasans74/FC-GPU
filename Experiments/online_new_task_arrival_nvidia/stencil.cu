#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h> // Use cuda_runtime.h for cudaMalloc, cudaHostAlloc, cudaEvent, etc.
#include <iostream>
#include <csignal>
#include <thread>
#include <chrono>
#include <future>
#include <vector>
#include <fstream> // For std::ofstream
#include <time.h>  // For clock_gettime
#include <errno.h> // For errno with shmget

// CUDA_CHECK macro for robust NVIDIA CUDA error handling
#define CUDA_CHECK(call)                                                         \
  do {                                                                           \
    cudaError_t err = call;                                                      \
    if (err != cudaSuccess) {                                                    \
      std::cerr << "CUDA Error: "                                                \
                << cudaGetErrorString(err)                                       \
                << " at " << __FILE__                                            \
                << ":" << __LINE__                                               \
                << " in " << #call << std::endl;                                 \
      exit(EXIT_FAILURE);                                                        \
    }                                                                            \
  } while (0)

#include "shared_data.h"

float period;
float setpoint;
float termination;
using namespace std;
int pids;
typedef double Real;
#define GLOBAL_IDX(x, y, z, nx_dim, ny_dim, nz_dim) ((z) + (nz_dim * ((y) + (ny_dim * (x)))))


#ifndef SIGNAL_TYPE
    #ifdef T1
        #define SIGNAL_TYPE SIGUSR1
        #define SHARED_MEM_INDEX 0
        #define SHARED_INDEX_FOR_PERIOD 0 // Renamed from SHARED_MEM_INDEX2 for clarity
        #define LOG_SUFFIX "1"
    #elif defined(T2)
        #define SIGNAL_TYPE SIGUSR2
        #define SHARED_MEM_INDEX 1
        #define SHARED_INDEX_FOR_PERIOD 1
        #define LOG_SUFFIX "2"
    #elif defined(T3)
        #define SIGNAL_TYPE SIGUSR3
        #define SHARED_MEM_INDEX 4
        #define SHARED_INDEX_FOR_PERIOD 2
        #define LOG_SUFFIX "3"
    #elif defined(T4)
        #define SIGNAL_TYPE SIGUSR4
        #define SHARED_MEM_INDEX 5
        #define SHARED_INDEX_FOR_PERIOD 3
        #define LOG_SUFFIX "4"
    #else
        #define SIGNAL_TYPE SIGUSR1
        #define SHARED_MEM_INDEX 0
        #define SHARED_INDEX_FOR_PERIOD 0
        #define LOG_SUFFIX "1"
    #endif
#endif

#define MAKE_LOG_PATH(prefix) ("logs/" prefix LOG_SUFFIX ".txt")

std::string outputFilePath = MAKE_LOG_PATH("log");
std::string slackLogPath = MAKE_LOG_PATH("s");
std::string periodLogPath = MAKE_LOG_PATH("p");
std::string rtjLogPath = MAKE_LOG_PATH("rtj");
std::string preemptionLogPath = MAKE_LOG_PATH("preemptiont");

std::ofstream outputfile(outputFilePath);
std::ofstream slack1(slackLogPath);
std::ofstream period1(periodLogPath);
std::ofstream rtj1(rtjLogPath);
std::ofstream preemptiontime(preemptionLogPath, std::ios::app);


#define BSIZE 16
#define XTILE 20


const int size = 512;
int repeat = 375;
const int nx = size, ny = size, nz = size;
const int vol = nx * ny * nz;

// Global host pointers (allocated with cudaHostAllocMapped)
Real *h_Vm, *h_dVm, *h_sigma;

// Global device pointers (aliases to the host pointers above, obtained via cudaHostGetDevicePointer)
Real *d_Vm_dev, *d_dVm_dev, *d_sigma_dev;


__global__ void stencil3d(
    const Real *__restrict__ d_psi_arg,
          Real *__restrict__ d_npsi_arg,
    const Real *__restrict__ d_sigmaX,
    const Real *__restrict__ d_sigmaY,
    const Real *__restrict__ d_sigmaZ,
    int nx_arg, int ny_arg, int nz_arg)
{
    
    __shared__ Real sm_psi[4][BSIZE][BSIZE]; // 4 planes to handle pii, cii, nii and a temporary swap

    const int tjj = threadIdx.y; // local y-index within block
    const int tkk = threadIdx.x; // local z-index within block

    // Indices for circular buffer in shared memory
    int pii = 0, cii = 1, nii = 2, tii;

    // Calculate the global base coordinates for this block's region
    int block_base_x = XTILE * blockIdx.x;
    int block_base_y = (BSIZE - 2) * blockIdx.y; // Y-base for the actual computation region
    int block_base_z = (BSIZE - 2) * blockIdx.z; // Z-base for the actual computation region

    // These are for the BSIZE x BSIZE shared memory region, which includes a halo if (BSIZE-2) is the compute region.
    int global_y_sm = block_base_y + tjj;
    int global_z_sm = block_base_z + tkk;

    if (global_y_sm >= ny_arg || global_z_sm >= nz_arg) {
        // Mark shared memory for this thread as invalid or handle appropriately
        // For simplicity here, just return for threads outside the global bounds.
        return;
    }

    // Determine the actual number of X-slices to process for this block, considering grid boundaries
    int current_block_xtile = XTILE;
    if (blockIdx.x == gridDim.x - 1) { // If it's the last block in X
        current_block_xtile = nx_arg - 2 - block_base_x; // Adjust for actual remaining X-slices
        if (current_block_xtile < 0) current_block_xtile = 0; // No slices left
    }


    // Adjust indices for X-planes (0 and 1 relative to block_base_x) and global Y,Z
    if (block_base_x < nx_arg -1) { // Check if global_x + 0 is within bounds (for 'current' plane)
        sm_psi[cii][tjj][tkk] = d_psi_arg[GLOBAL_IDX(block_base_x + 0, global_y_sm, global_z_sm, nx_arg, ny_arg, nz_arg)];
    } else {
        sm_psi[cii][tjj][tkk] = 0.0; // Or some boundary value
    }

    if (block_base_x + 1 < nx_arg -1) { // Check if global_x + 1 is within bounds (for 'next' plane)
        sm_psi[nii][tjj][tkk] = d_psi_arg[GLOBAL_IDX(block_base_x + 1, global_y_sm, global_z_sm, nx_arg, ny_arg, nz_arg)];
    } else {
        sm_psi[nii][tjj][tkk] = 0.0; // Or some boundary value
    }
    
    __syncthreads(); // Ensure initial shared memory loads are complete

    // The 'ii' variable effectively represents the relative x-offset from block_base_x
    for (int ii = 0; ii < current_block_xtile; ii++) {
        // Before computation, load the (ii+2)-th plane (relative to block_base_x) into shared memory
        // This will become the 'nii' plane for the *next* iteration.
        int next_global_x = block_base_x + ii + 2;
        if (next_global_x < nx_arg -1) { // Check boundary for the next plane to load
            sm_psi[3][tjj][tkk] = d_psi_arg[GLOBAL_IDX(next_global_x, global_y_sm, global_z_sm, nx_arg, ny_arg, nz_arg)];
        } else {
            sm_psi[3][tjj][tkk] = 0.0; // Boundary value
        }
        __syncthreads(); // Ensure next plane is loaded before computation starts on current plane

  
        if (tjj > 0 && tjj < BSIZE - 1 && tkk > 0 && tkk < BSIZE - 1) {
            // This is just a placeholder example, replace with your actual stencil equation
            Real current_val = sm_psi[cii][tjj][tkk];
            Real neighbor_x_prev = sm_psi[pii][tjj][tkk];
            Real neighbor_x_next = sm_psi[nii][tjj][tkk];
            Real neighbor_y_prev = sm_psi[cii][tjj - 1][tkk];
            Real neighbor_y_next = sm_psi[cii][tjj + 1][tkk];
            Real neighbor_z_prev = sm_psi[cii][tjj][tkk - 1];
            Real neighbor_z_next = sm_psi[cii][tjj][tkk + 1];

            Real sigma_val_x = d_sigmaX[GLOBAL_IDX(block_base_x + ii + 1, global_y_sm, global_z_sm, nx_arg, ny_arg, nz_arg)]; // Example sigma access
            Real sigma_val_y = d_sigmaY[GLOBAL_IDX(block_base_x + ii + 1, global_y_sm, global_z_sm, nx_arg, ny_arg, nz_arg)];
            Real sigma_val_z = d_sigmaZ[GLOBAL_IDX(block_base_x + ii + 1, global_y_sm, global_z_sm, nx_arg, ny_arg, nz_arg)];

            Real result = current_val; // Placeholder
            int output_global_x = block_base_x + ii + 1;
            // Only write if within valid output bounds (not halo regions)
            if (output_global_x >= 1 && output_global_x < nx_arg - 1 &&
                global_y_sm >= 1 && global_y_sm < ny_arg - 1 &&
                global_z_sm >= 1 && global_z_sm < nz_arg - 1) {
                
                d_npsi_arg[GLOBAL_IDX(output_global_x, global_y_sm, global_z_sm, nx_arg, ny_arg, nz_arg)] = result;
            }
        }

        __syncthreads();
        tii = pii;
        pii = cii;
        cii = nii;
        nii = 3; // 'nii' for the next iteration will be the plane just loaded (index 3)
    }
}

void init()
{
    FILE *file = fopen("logs/mainpid.txt", "r");
    int value;
    if (file) {
        if (fscanf(file, "%d", &value) == 1) {
            pids = value;
            fclose(file);
            std::cout << "PID PARENT T" << LOG_SUFFIX << "= " << pids << endl;
        } else {
            std::cerr << "Error reading PID from logs/mainpid.txt." << std::endl;
            pids = 0;
            fclose(file);
        }
    } else {
        std::cerr << "Error: logs/mainpid.txt not found. Cannot determine parent PID." << std::endl;
        pids = 0;
    }
}
int jobs = 2;
bool keepRunning = true;
bool signalReceived_global = false; // Renamed to avoid local variable shadowing
void handleSignal(int signal) {
    if (signal == SIGHUP) {
        ::signalReceived_global = true; // Set the global flag
        ::keepRunning = false; // Also set keepRunning to false for graceful exit
        std::cout << "SIGHUP received, initiating graceful shutdown." << std::endl;
    }
}
  

std::vector<float> responsetime;
float min_rate;
float max_rate;

float calculateaverage(std::vector<float> resp_time)
{
    if (resp_time.empty()) {
        return 0.0f;
    }
    float avg_resp_time=0;
    for (const auto& time : resp_time) {
        avg_resp_time += time;
    }
    avg_resp_time /= resp_time.size();
    return avg_resp_time;
}
std::chrono::time_point<std::chrono::steady_clock> controltime;
std::chrono::time_point<std::chrono::steady_clock> program_start_time_global; // Renamed to avoid local variable shadowing
std::chrono::time_point<std::chrono::steady_clock> preemptionlaunch;


cudaEvent_t start, stop;
struct timespec starta, enda;

// Initialize CUDA events for timing
void initializeCudaEvents() {
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
}

// kernellaunch now uses the device-mapped pointers directly
void kernellaunch(dim3 gridDim, dim3 blockDim) {
    auto preemption = std::chrono::steady_clock::now();
    auto preemption_start_duration_chrono = std::chrono::duration_cast<std::chrono::milliseconds>(preemption - preemptionlaunch).count();
    preemptiontime << "Preemption time = " << preemption_start_duration_chrono << " ms" << std::endl;
    clock_gettime(CLOCK_MONOTONIC_RAW, &enda);
    double host_preemption_ms = (enda.tv_sec - starta.tv_sec) * 1e3 + (enda.tv_nsec - starta.tv_nsec) / 1e6; // in milliseconds
    float kernel_elapsed_time_ms = 0.0f;
    float total_kernel_time_sum_ms = 0.0f;

    for (int i = 0; i < jobs; ++i) {
        CUDA_CHECK(cudaEventRecord(start));

        // Launch the kernel multiple times based on the repeat count
        for (int j = 0; j < repeat; ++j) {
            // Pass the device-mapped pointers to the kernel
            stencil3d<<<gridDim, blockDim>>>(d_Vm_dev, d_dVm_dev, d_sigma_dev, d_sigma_dev + 3 * vol, d_sigma_dev + 6 * vol, nx, ny, nz);
        }

        // Synchronize to ensure all kernels are completed
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        CUDA_CHECK(cudaEventElapsedTime(&kernel_elapsed_time_ms, start, stop));
        total_kernel_time_sum_ms += kernel_elapsed_time_ms;

        outputfile << "Job time = \t" << kernel_elapsed_time_ms << " ms" << std::endl;
    }

    // Calculate and store the average response time (kernel time + host preemption)
    float avg_response_time_ms = (total_kernel_time_sum_ms / jobs) + host_preemption_ms;
    sharedData->values[SHARED_MEM_INDEX] = avg_response_time_ms / 1.0f; // Convert to seconds
    responsetime.push_back(avg_response_time_ms); // Add to vector for overall average
}

void runService(float setpoint_arg, float period_arg, float termination_arg) {
    program_start_time_global = std::chrono::steady_clock::now();
    controltime = std::chrono::steady_clock::now();

    printf("Grid dimension: nx=%d ny=%d nz=%d\n", nx, ny, nz);

    // Allocate memory on host (pinned/page-locked) that is also directly accessible by the device (zero-copy)
    CUDA_CHECK(cudaHostAlloc((void **)&h_Vm, sizeof(Real) * vol, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc((void **)&h_sigma, sizeof(Real) * vol * 9, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc((void **)&h_dVm, sizeof(Real) * vol, cudaHostAllocMapped));

    // Get the corresponding device pointers (aliases to the host pointers)
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&d_Vm_dev, h_Vm, 0));
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&d_sigma_dev, h_sigma, 0));
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&d_dVm_dev, h_dVm, 0));

    // Initialize host memory (h_Vm, h_sigma, h_dVm) directly.
    // This data is immediately visible to the GPU via d_Vm_dev, d_sigma_dev, d_dVm_dev.
    for (int i = 0; i < vol; ++i) h_Vm[i] = i % 19;
    for (int i = 0; i < vol * 9; ++i) h_sigma[i] = i % 19;
    CUDA_CHECK(cudaMemset(h_dVm, 0, sizeof(Real) * vol)); // Initialize h_dVm to zero

    // Compute grid and block dimensions
    int bdimx = (nx - 2) / XTILE + ((nx - 2) % XTILE == 0 ? 0 : 1);
    int bdimy = (ny - 2) / (BSIZE - 2) + ((ny - 2) % (BSIZE - 2) == 0 ? 0 : 1);
    int bdimz = (nz - 2) / (BSIZE - 2) + ((nz - 2) % (BSIZE - 2) == 0 ? 0 : 1);

    dim3 grids(bdimx, bdimy, bdimz);
    dim3 blocks(BSIZE, BSIZE, 1);

    std::future<void> asyncTask;
    std::vector<float> controlperiod_log;
   
    //init(); // Uncomment if you need parent PID for signals
    //kill(pids, SIGNAL_TYPE); // Uncomment if you need to send initial signal
    sharedData->newperiods[SHARED_INDEX_FOR_PERIOD] = period_arg;
    initializeCudaEvents(); 

    while (keepRunning) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(now - program_start_time_global).count() / 1000.0;
        outputfile << "Task launched at: " << elapsed_time << " sec" << std::endl << std::endl;

        if (elapsed_time > termination_arg) {
            std::cout << "Termination time reached. Exiting." << std::endl;
            break;
        }
        if (!::keepRunning) { // Check global signal flag (set by handleSignal)
            std::cout << "Shutdown signal received. Terminating gracefully." << std::endl;
            break;
        }

        preemptionlaunch = std::chrono::steady_clock::now();
        clock_gettime(CLOCK_MONOTONIC_RAW, &starta);

        if (asyncTask.valid()) {
            asyncTask.get();
            float avg_resp_time = calculateaverage(responsetime);
            outputfile << "Average response time: " << avg_resp_time << " ms" << std::endl;
            rtj1 << avg_resp_time << std::endl;
            responsetime.clear();
            controlperiod_log.push_back(avg_resp_time);
        }

        // Launch the asynchronous task
        asyncTask = std::async(std::launch::async, [&]() {
            kernellaunch(grids, blocks); // No h_Vm, h_sigma needed as arguments
        });

        period1 << sharedData->newperiods[SHARED_INDEX_FOR_PERIOD] << std::endl;

        // Sleep for the specified period
        std::chrono::milliseconds sleep_duration(static_cast<long long>(sharedData->newperiods[SHARED_INDEX_FOR_PERIOD] * 1000.0));
        std::this_thread::sleep_for(sleep_duration);
    }

    if (asyncTask.valid()) {
        asyncTask.get();
    }

    // Cleanup resources
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    // Free host-mapped memory using cudaFreeHost
    CUDA_CHECK(cudaFreeHost(h_Vm));
    CUDA_CHECK(cudaFreeHost(h_sigma));
    CUDA_CHECK(cudaFreeHost(h_dVm));

    slack1.close();
    period1.close();
    rtj1.close();
    outputfile.close();
    preemptiontime.close();
}


int main(int argc, char *argv[]) {
     if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <setpoint> <period> <termination>" << std::endl;
        return 1;
    }

    setpoint = atof(argv[1]);
    period = atof(argv[2]);
    termination = atof(argv[3]);
    min_rate=period-period*0.90;
    max_rate=period+period*4.90;
    std::signal(SIGHUP, handleSignal);
    
    key_t key = ftok("shmfile", 65);
    int shmid = shmget(key, 1024, 0666 | IPC_CREAT);
    bool createdshm;
    if(shmid>0){
        sharedData = (SharedData*)shmat(shmid, nullptr, 0);
        createdshm=true;
    }
    else
    {
        shmid = shmget(key, sizeof(SharedData), 0666);
        sharedData = (SharedData*)shmat(shmid, nullptr, 0);
    }

    std::cout<<"Min rate = \t"<<min_rate<<"Max rate = "<<max_rate<<std::endl;
    std::cout << "Service is running. Press Ctrl+C to exit." << std::endl;
    runService(setpoint, period, termination);
    outputfile.close();
    std::cout << "Service is shutting down." << std::endl;
    
    
   if(createdshm == true)
    {
        shmdt(sharedData);
        shmctl(shmid,  IPC_RMID, NULL);

    }

    return 0;
}
