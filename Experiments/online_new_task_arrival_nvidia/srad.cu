#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <iostream>
#include <csignal>
#include <thread>
#include <chrono>
#include <future>
#include <vector>
#include <fstream> // For std::ofstream
#include <time.h>  // For clock_gettime
#include <errno.h> // For errno with shmget
#include "shared_data.h"

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


// Global variables for period, setpoint, termination
float period;
float setpoint;
float termination;
using namespace std;
int pids;
int jobs = 2; // Number of kernel repetitions per cycle
int size_I;   // Total elements in the image (rows * cols)

#ifndef SIGNAL_TYPE
    #ifdef T1
        #define SIGNAL_TYPE SIGUSR1
        #define SHARED_MEM_INDEX 0
        #define SHARED_INDEX_FOR_PERIOD 0
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


#ifdef RD_WG_SIZE_0_0
    #define BLOCK_SIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
    #define BLOCK_SIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
    #define BLOCK_SIZE RD_WG_SIZE
#else
    #define BLOCK_SIZE 16
#endif

// Global host pointers (will be zero-copy mapped)
float *h_I, *h_J;
float *h_C, *h_E, *h_W, *h_N, *h_S;

// Global device pointers (aliases to the host pointers above)
// These are the pointers you'll pass to the kernels.
float *d_J, *d_C, *d_E, *d_W, *d_N, *d_S;

int rows = 512 * 2;
int cols = 512 * 2;
float lambda = 2.5;
float q0sqr; // Will be calculated based on input image
// Kernel 1: SRAD CUDA computation
__global__ void srad_cuda_1(
    float *E_C, float *W_C, float *N_C, float *S_C,
    float *J_cuda, float *C_cuda, int cols, int rows, float q0sqr) 
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int index = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + tx;

    __shared__ float temp[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float temp_result[BLOCK_SIZE][BLOCK_SIZE];

    temp[ty][tx] = J_cuda[index];
    __syncthreads();

    float jc = temp[ty][tx];
    float n = (ty == 0) ? jc : temp[ty - 1][tx] - jc;
    float s = (ty == BLOCK_SIZE - 1) ? jc : temp[ty + 1][tx] - jc;
    float w = (tx == 0) ? jc : temp[ty][tx - 1] - jc;
    float e = (tx == BLOCK_SIZE - 1) ? jc : temp[ty][tx + 1] - jc;

    float g2 = (n * n + s * s + w * w + e * e) / (jc * jc);
    float l = (n + s + w + e) / jc;
    float num = (0.5f * g2) - ((1.0f / 16.0f) * (l * l));
    float den = 1 + (0.25f * l);
    float qsqr = num / (den * den);
    den = (qsqr - q0sqr) / (q0sqr * (1 + q0sqr));
    float c = 1.0f / (1.0f + den);

    temp_result[ty][tx] = fminf(fmaxf(c, 0.0f), 1.0f);
    __syncthreads();

    C_cuda[index] = temp_result[ty][tx];
    E_C[index] = e;
    W_C[index] = w;
    N_C[index] = n;
    S_C[index] = s;
}

// Kernel 2: SRAD update
__global__ void srad_cuda_2(
    float *E_C, float *W_C, float *N_C, float *S_C,
    float *J_cuda, float *C_cuda, int cols, int rows, float lambda) 
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int index = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + tx;

    __shared__ float temp[BLOCK_SIZE][BLOCK_SIZE];
    temp[ty][tx] = J_cuda[index];
    __syncthreads();

    float cc = C_cuda[index];
    float d_sum = cc * (N_C[index] + S_C[index] + W_C[index] + E_C[index]);
    temp[ty][tx] += 0.25f * lambda * d_sum;
    __syncthreads();

    J_cuda[index] = temp[ty][tx];
}


// Function to generate a random matrix
void random_matrix(float *matrix, int rows_arg, int cols_arg) {
    srand(7); // Fixed seed for reproducibility
    for (int i = 0; i < rows_arg; i++) {
        for (int j = 0; j < cols_arg; j++) {
            matrix[i * cols_arg + j] = rand() / (float)RAND_MAX;
        }
    }
}


cudaEvent_t start, stop;
struct timespec starta, enda;

// Initialize CUDA events for timing
void initializeCudaEvents() {
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
}


void kernellaunch(dim3 gridDim, dim3 blockDim)
{
    auto preemption = std::chrono::steady_clock::now();
    auto preemption_start_duration_chrono = std::chrono::duration_cast<std::chrono::milliseconds>(preemption - preemptionlaunch).count();
    preemptiontime << "Preemption time = " << preemption_start_duration_chrono << " ms" << std::endl;

    // Host-side preemption measurement (starta set before asyncTask launch)
    clock_gettime(CLOCK_MONOTONIC_RAW, &enda);
    double host_preemption_ms = (enda.tv_sec - starta.tv_sec) * 1e3 + (enda.tv_nsec - starta.tv_nsec) / 1e6;

    float kernel_elapsed_time_ms = 0.0f;  // Time for one full iteration (both kernels)
    float total_kernel_time_sum_ms = 0.0f; // Sum of kernel times over all 'jobs' repetitions

    for (int i = 0; i < jobs; ++i) {
        CUDA_CHECK(cudaEventRecord(start));

        // Launch the kernels using the device pointers
        srad_cuda_1<<<gridDim, blockDim>>>(d_E, d_W, d_N, d_S, d_J, d_C, cols, rows, q0sqr);
        srad_cuda_2<<<gridDim, blockDim>>>(d_E, d_W, d_N, d_S, d_J, d_C, cols, rows, lambda);

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&kernel_elapsed_time_ms, start, stop));

        total_kernel_time_sum_ms += kernel_elapsed_time_ms;
        outputfile << "Job time (iteration " << i+1 << "): " << kernel_elapsed_time_ms << " ms" << std::endl;
    }

    // Calculate and store the average response time (kernel time + host preemption)
    float avg_response_time_ms = (total_kernel_time_sum_ms / jobs) + host_preemption_ms;
    sharedData->values[SHARED_MEM_INDEX] = avg_response_time_ms;
    responsetime.push_back(avg_response_time_ms); // Add to vector for overall average
}


void runService(float setpoint_arg, float period_arg, float termination_arg) {
    program_start_time_global = std::chrono::steady_clock::now();
    controltime = std::chrono::steady_clock::now(); // Not actively used but kept from original

    if (rows % BLOCK_SIZE != 0 || cols % BLOCK_SIZE != 0) {
        fprintf(stderr, "Error: rows (%d) and cols (%d) must be multiples of BLOCK_SIZE (%d)\n", rows, cols, BLOCK_SIZE);
        exit(EXIT_FAILURE);
    }

    size_I = rows * cols; // Initialize global size_I

    // Allocate host memory (pinned/page-locked) that is also directly accessible by the device (zero-copy)
    CUDA_CHECK(cudaHostAlloc((void**)&h_I, size_I * sizeof(float), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc((void**)&h_J, size_I * sizeof(float), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc((void**)&h_C, size_I * sizeof(float), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc((void**)&h_E, size_I * sizeof(float), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc((void**)&h_W, size_I * sizeof(float), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc((void**)&h_N, size_I * sizeof(float), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc((void**)&h_S, size_I * sizeof(float), cudaHostAllocMapped));

    // Get the corresponding device pointers (aliases to the host pointers)
    // These are the pointers you will pass to the kernels.
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&d_J, h_J, 0));
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&d_C, h_C, 0));
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&d_E, h_E, 0));
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&d_W, h_W, 0));
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&d_N, h_N, 0));
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&d_S, h_S, 0));

    // Initialize host memory (h_I, h_J) which is directly visible to GPU via d_J
    random_matrix(h_I, rows, cols); // Populate h_I with random data
    for (int i = 0; i < size_I; i++) {
        h_J[i] = expf(h_I[i]); // Initialize h_J based on h_I
    }

    // Calculate q0sqr based on the initial J image
    // Find min/max values in J to calculate q0
    float min_val = h_J[0];
    float max_val = h_J[0];
    for (int i = 1; i < size_I; ++i) {
        if (h_J[i] < min_val) min_val = h_J[i];
        if (h_J[i] > max_val) max_val = h_J[i];
    }
    float q0 = 1.0f - (min_val / max_val); // This is a common way to calculate q0 for SRAD
    q0sqr = q0 * q0; // Set the global q0sqr

    std::future<void> asyncTask;
    std::vector<float> controlperiod_log; // Renamed to avoid local variable shadowing
    initializeCudaEvents();

    // init(); // Uncomment if you need parent PID for signals
    // if (pids != 0) kill(pids, SIGNAL_TYPE); // Example: send initial signal if needed

    sharedData->newperiods[SHARED_INDEX_FOR_PERIOD] = period_arg; // Set initial period in shared memory

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(cols / BLOCK_SIZE, rows / BLOCK_SIZE);
   
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
            asyncTask.get(); // Wait for previous task to finish
            float avg_resp_time = calculateaverage(responsetime);
            outputfile << "Average response time (last cycle): " << avg_resp_time << " ms" << std::endl;
            rtj1 << avg_resp_time << std::endl;
            responsetime.clear();
            controlperiod_log.push_back(avg_resp_time);
        }
        
        asyncTask = std::async(std::launch::async, [&]() {
            kernellaunch(dimGrid, dimBlock);
        });
        
        period1 << sharedData->newperiods[SHARED_INDEX_FOR_PERIOD] << std::endl;
   
        std::chrono::milliseconds sleep_duration(static_cast<long long>(sharedData->newperiods[SHARED_INDEX_FOR_PERIOD] * 1000.0));
        std::this_thread::sleep_for(sleep_duration);
    }

    if (asyncTask.valid()) {
        asyncTask.get(); // Ensure the last asynchronous task completes
    }

    printf("Computation Done\n");
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Free all host-mapped memory using cudaFreeHost
    CUDA_CHECK(cudaFreeHost(h_I));
    CUDA_CHECK(cudaFreeHost(h_J));
    CUDA_CHECK(cudaFreeHost(h_C));
    CUDA_CHECK(cudaFreeHost(h_E));
    CUDA_CHECK(cudaFreeHost(h_W));
    CUDA_CHECK(cudaFreeHost(h_N));
    CUDA_CHECK(cudaFreeHost(h_S));

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
    std::signal(SIGHUP, handleSignal);
    
    key_t key = ftok("shmfile", 65);
    int shmid = shmget(key, sizeof(SharedData), 0666 | IPC_CREAT); // Use sizeof(SharedData)
    bool createdshm_by_this_process = false;

    if(shmid == -1){
        if (errno == EEXIST) { // Shared memory segment already exists
            shmid = shmget(key, sizeof(SharedData), 0666);
            if (shmid == -1) {
                perror("shmget failed to get existing shared memory");
                return 1;
            }
        } else { // Other shmget error
            perror("shmget failed for creation");
            return 1;
        }
    } else {
        createdshm_by_this_process = true;
    }

    sharedData = (SharedData*)shmat(shmid, nullptr, 0);
    if(sharedData == (void*)-1) {
        perror("shmat failed");
        if (createdshm_by_this_process) {
             shmctl(shmid, IPC_RMID, NULL);
        }
        return 1;
    }

    // Initialize shared memory if this process created it
    if (createdshm_by_this_process) {
        memset(sharedData, 0, sizeof(SharedData)); // Zero out the entire shared memory segment
        sharedData->values[SHARED_MEM_INDEX] = 0.0f;
        sharedData->newperiods[SHARED_INDEX_FOR_PERIOD] = period; // Initial period from cmd line
    }

    std::cout<<"Min rate = \t"<<min_rate<<"Max rate = "<<max_rate<<std::endl;
    std::cout << "Service is running. Press Ctrl+C to exit." << std::endl;
    runService(setpoint, period, termination); // Pass arguments to runService
    
    std::cout << "Service is shutting down." << std::endl;
    
    // Cleanup shared memory
    if(sharedData != (void*)-1) {
        if (shmdt(sharedData) == -1) {
            perror("shmdt failed");
        }
    }

    if(createdshm_by_this_process) {
        if (shmctl(shmid, IPC_RMID, NULL) == -1) {
            perror("shmctl IPC_RMID failed");
        }
    }

    return 0;
}
