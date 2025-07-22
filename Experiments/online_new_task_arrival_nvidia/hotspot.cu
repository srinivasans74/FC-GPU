#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <signal.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <future>
#include <cuda_runtime.h> // Correct header for CUDA runtime API functions
#include <assert.h>
#include <string.h> // For memcpy
#include <errno.h>  // For errno with shmget

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

#include"shared_data.h"

float period;
float setpoint;
float termination;
using namespace std;
int pids;
// int jobs = 2; // Moved to global scope below for clarity and consistency


#ifndef SIGNAL_TYPE
    #ifdef T1
        #define SIGNAL_TYPE SIGUSR1  // For T1: SIGUSR1
        #define SHARED_MEM_INDEX 0   // Index: 0
        #define SHARED_INDEX_FOR_PERIOD 0  // Index for periods in shared memory
        #define LOG_SUFFIX "1"   // Default suffix if no flag is provided
    #elif defined(T2)
        #define SIGNAL_TYPE SIGUSR2  // For T2: SIGUSR2
        #define SHARED_MEM_INDEX 1   // Index: 1
        #define SHARED_INDEX_FOR_PERIOD 1  // Index for periods in shared memory
        #define LOG_SUFFIX "2"   // Default suffix if no flag is provided
    #elif defined(T3)
        #define SIGNAL_TYPE SIGUSR3  // For T3: SIGUSR3
        #define SHARED_MEM_INDEX 4   // Index: 4
        #define SHARED_INDEX_FOR_PERIOD 2  // Index for periods in shared memory
        #define LOG_SUFFIX "3"   // Default suffix if no flag is provided
    #elif defined(T4)
        #define SIGNAL_TYPE SIGUSR4  // For T4: SIGUSR4
        #define SHARED_MEM_INDEX 5   // Index: 5
        #define SHARED_INDEX_FOR_PERIOD 3  // Index for periods in shared memory
        #define LOG_SUFFIX "4"   // Default suffix if no flag is provided
    #else
        #define SIGNAL_TYPE SIGUSR1  // Default: SIGUSR1
        #define SHARED_MEM_INDEX 0   // Default Index: 0
        #define SHARED_INDEX_FOR_PERIOD 0  // Default Index for periods in shared memory
        #define LOG_SUFFIX "1"   // Default suffix if no flag is provided
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


#ifdef RD_WG_SIZE_0_0
    #define BLOCK_SIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
    #define BLOCK_SIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
    #define BLOCK_SIZE RD_WG_SIZE
#else
    #define BLOCK_SIZE 16
#endif

#define STR_SIZE 256
#define MAX_PD 3.0e6
#define PRECISION 0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
#define FACTOR_CHIP 0.5
#define HALO 1

float t_chip = 0.0005;
float chip_height = 0.016;
float chip_width = 0.016;
float amb_temp = 80.0;

// Global variables (already present)
// float period; // Already declared above
// float setpoint; // Already declared above
// float termination; // Already declared above
float min_rate;
float max_rate;
int rows, cols;
int* data; // Not used in this version of the code
int** wall; // Not used
int* result; // Not used
int pyramid_height; // Needs to be initialized, e.g., to 1
bool keepRunning = true;

std::vector<float> controlperiod;
std::vector<float> responsetime;

std::chrono::time_point<std::chrono::steady_clock> controltime;
std::chrono::time_point<std::chrono::steady_clock> program_start_time;
std::chrono::time_point<std::chrono::steady_clock> preemptionlaunch;


#define IN_RANGE(x, min, max) ((x) >= (min) && (x) <= (max))
#define CLAMP_RANGE(x, min, max) x = ((x) < (min)) ? (min) : (((x) > (max)) ? (max) : (x))
#define MIN(a, b) ((a) <= (b) ? (a) : (b))

// Signal handler
void handleSignal(int signal) {
    if (signal == SIGHUP) {
        ::keepRunning = false; // Set the global flag to false for graceful exit
        std::cout << "SIGHUP received, initiating graceful shutdown." << std::endl;
    }
}

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
// using namespace std; // Already done at top
// int pids; // Already declared above

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

void fatal(const char *s) {
    fprintf(stderr, "error: %s\n", s);
    exit(EXIT_FAILURE);
}

void readinput(float *vect, int grid_rows, int grid_cols, char *file){
    int i,j;
    FILE *fp;
    char str[STR_SIZE];
    float val;

    if( (fp  = fopen(file, "r" )) ==0 )
            fatal("The file was not opened"); // Use fatal for consistent error handling

    for (i=0; i <= grid_rows-1; i++)
     for (j=0; j <= grid_cols-1; j++)
     {
        fgets(str, STR_SIZE, fp);
        if (feof(fp))
            fatal("not enough lines in file");
        if ((sscanf(str, "%f", &val) != 1))
            fatal("invalid file format");
        vect[i*grid_cols+j] = val;
    }
    fclose(fp);    
}

// KERNEL SIGNATURE MODIFIED TO ACCEPT DEVICE POINTERS
__global__ void calculate_temp(int iteration,  //number of iteration
                               float *d_power,    //power input (device pointer)
                               float *d_temp_src,     //temperature input/output (device pointer)
                               float *d_temp_dst,     //temperature input/output (device pointer)
                               int grid_cols,  //Col of grid
                               int grid_rows,  //Row of grid
                               int border_cols,    // border offset 
                               int border_rows,    // border offset
                               float Cap,       //Capacitance
                               float Rx,   
                               float Ry,   
                               float Rz,   
                               float step, 
                               float time_elapsed){
    
        __shared__ float temp_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float power_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float temp_t[BLOCK_SIZE][BLOCK_SIZE]; // saving temparary temperature result

    float amb_temp_local = 80.0; // Using a local variable to avoid global access in kernel
        float step_div_Cap;
        float Rx_1,Ry_1,Rz_1;
        
    int bx = blockIdx.x;
        int by = blockIdx.y;

    int tx=threadIdx.x;
    int ty=threadIdx.y;
    
    step_div_Cap=step/Cap;
    
    Rx_1=1/Rx;
    Ry_1=1/Ry;
    Rz_1=1/Rz;
    
        // each block finally computes result for a small block
        // after N iterations. 
        // it is the non-overlapping small blocks that cover 
        // all the input data

        // calculate the small block size
    int small_block_rows = BLOCK_SIZE-iteration*2;//EXPAND_RATE
    int small_block_cols = BLOCK_SIZE-iteration*2;//EXPAND_RATE

        // calculate the boundary for the block according to 
        // the boundary of its small block
        int blkY = small_block_rows*by-border_rows;
        int blkX = small_block_cols*bx-border_cols;
        int blkYmax = blkY+BLOCK_SIZE-1;
        int blkXmax = blkX+BLOCK_SIZE-1;

        // calculate the global thread coordination
    int yidx = blkY+ty;
    int xidx = blkX+tx;

        // load data if it is within the valid input range
    int loadYidx=yidx, loadXidx=xidx;
        int index = grid_cols*loadYidx+loadXidx;
        
    if(IN_RANGE(loadYidx, 0, grid_rows-1) && IN_RANGE(loadXidx, 0, grid_cols-1)){
            temp_on_cuda[ty][tx] = d_temp_src[index];  // Load from device pointer
            power_on_cuda[ty][tx] = d_power[index];// Load from device pointer
    }
    __syncthreads();

        // effective range within this block that falls within 
        // the valid range of the input data
        // used to rule out computation outside the boundary.
        int validYmin = (blkY < 0) ? -blkY : 0;
        int validYmax = (blkYmax > grid_rows-1) ? BLOCK_SIZE-1-(blkYmax-grid_rows+1) : BLOCK_SIZE-1;
        int validXmin = (blkX < 0) ? -blkX : 0;
        int validXmax = (blkXmax > grid_cols-1) ? BLOCK_SIZE-1-(blkXmax-grid_cols+1) : BLOCK_SIZE-1;

        int N = ty-1;
        int S = ty+1;
        int W = tx-1;
        int E = tx+1;
        
        N = (N < validYmin) ? validYmin : N;
        S = (S > validYmax) ? validYmax : S;
        W = (W < validXmin) ? validXmin : W;
        E = (E > validXmax) ? validXmax : E;

        bool computed;
        for (int i=0; i<iteration ; i++){ 
            computed = false;
            if( IN_RANGE(tx, i+1, BLOCK_SIZE-i-2) &&  \
                 IN_RANGE(ty, i+1, BLOCK_SIZE-i-2) &&  \
                 IN_RANGE(tx, validXmin, validXmax) && \
                 IN_RANGE(ty, validYmin, validYmax) ) {
                    computed = true;
                    temp_t[ty][tx] =    temp_on_cuda[ty][tx] + step_div_Cap * (power_on_cuda[ty][tx] + 
                    (temp_on_cuda[S][tx] + temp_on_cuda[N][tx] - 2.0*temp_on_cuda[ty][tx]) * Ry_1 + 
                    (temp_on_cuda[ty][E] + temp_on_cuda[ty][W] - 2.0*temp_on_cuda[ty][tx]) * Rx_1 + 
                    (amb_temp_local - temp_on_cuda[ty][tx]) * Rz_1); // Use local amb_temp
            }
            __syncthreads();
            if(i==iteration-1)
                break;
            if(computed)     //Assign the computation range
                temp_on_cuda[ty][tx]= temp_t[ty][tx];
            __syncthreads();
          }

      // update the global memory
      // after the last iteration, only threads coordinated within the 
      // small block perform the calculation and switch on ``computed''
      if (computed){
          d_temp_dst[index]= temp_t[ty][tx];        // Write to device pointer
      }
}



// Global host pointers (will be zero-copy mapped)
float *MatrixTemp[2], *MatrixPower;
float *d_MatrixTemp[2], *d_MatrixPower;

cudaEvent_t start, stop;
struct timespec starta, enda;

// Initialize CUDA events for timing
void initializeCudaEvents() {
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
}
int size; // Total elements in the grid (rows * cols)
float *FilesavingTemp, *FilesavingPower, *MatrixOut;
int jobs=2; // Number of kernel repetitions per cycle


// This is the `compute_tran_temp` called by `asyncTask`
// Its signature is modified to accept device pointers
void compute_tran_temp(float *d_MatrixPower_arg, float *d_MatrixTemp_arg[2], int col, int row,
                       int total_iterations, int num_iterations, int blockCols, int blockRows,
                       int borderCols, int borderRows) {
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(blockCols, blockRows);
    auto preemption_start_duration_chrono = std::chrono::duration_cast<std::chrono::milliseconds>(
                                                std::chrono::steady_clock::now() - preemptionlaunch)
                                                .count();
    preemptiontime << "Preemption time = " << preemption_start_duration_chrono / (1000.0) << " ms" << std::endl;
    clock_gettime(CLOCK_MONOTONIC_RAW, &enda);
    double host_preemption_ms = (enda.tv_sec - starta.tv_sec) * 1e3 + (enda.tv_nsec - starta.tv_nsec) / 1e6;

    float grid_height = chip_height / row;
    float grid_width = chip_width / col;

    float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
    float Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
    float Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
    float Rz = t_chip / (K_SI * grid_height * grid_width);

    float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    float step = PRECISION / max_slope;
    float time_elapsed = 0.001;

    float milliseconds = 0;
    float repsum = 0;

    for (int i = 0; i < jobs; i++) {
        CUDA_CHECK(cudaEventRecord(start));

        for (int t = 0; t < total_iterations; t += num_iterations) {
            int src = t % 2;
            int dst = 1 - src;

            // Launch kernel using the device pointers passed as arguments
            calculate_temp<<<dimGrid, dimBlock>>>(MIN(num_iterations, total_iterations - t),
                                                  d_MatrixPower_arg, d_MatrixTemp_arg[src], d_MatrixTemp_arg[dst],
                                                  col, row, borderCols, borderRows,
                                                  Cap, Rx, Ry, Rz, step, time_elapsed);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaGetLastError()); // Check for any asynchronous errors
        }

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        
        repsum += milliseconds;
        outputfile << "Job time for iteration " << i << ": " << milliseconds << " ms" << std::endl;
    }

    // Calculate and store the average response time (kernel time + host preemption)
    float avg_response_time_ms = (repsum / jobs) + host_preemption_ms;
    responsetime.push_back(avg_response_time_ms); // Add to vector for overall average
    sharedData->values[SHARED_MEM_INDEX] = avg_response_time_ms / 1.0f; // Convert to seconds
}


void run(float setpoint_arg, float period_arg, float termination_arg) {
    int grid_rows_val, grid_cols_val; // Use distinct names to avoid confusion with global 'rows', 'cols'
    const char *tfile_name, *pfile_name;

    int total_iterations = 10000;
    grid_rows_val = 1024*1;
    grid_cols_val = 1024*1;

    tfile_name = "temp_1024";
    pfile_name = "power_1024";

    size = grid_rows_val * grid_cols_val; // Initialize global 'size'

    pyramid_height = 1; // Initialize pyramid_height (was uninitialized)
    #define EXPAND_RATE 2 // Already defined, but good to note its usage
    int borderCols = (pyramid_height) * EXPAND_RATE / 2;
    int borderRows = (pyramid_height) * EXPAND_RATE / 2;
    int smallBlockCol = BLOCK_SIZE - (pyramid_height) * EXPAND_RATE;
    int smallBlockRow = BLOCK_SIZE - (pyramid_height) * EXPAND_RATE;
    // Ensure smallBlockCol/Row are positive and non-zero
    if (smallBlockCol <= 0) smallBlockCol = 1;
    if (smallBlockRow <= 0) smallBlockRow = 1;

    int blockCols = grid_cols_val / smallBlockCol + ((grid_cols_val % smallBlockCol == 0) ? 0 : 1);
    int blockRows = grid_rows_val / smallBlockRow + ((grid_rows_val % smallBlockRow == 0) ? 0 : 1);

    FilesavingTemp = (float*)malloc(size * sizeof(float));
    FilesavingPower = (float*)malloc(size * sizeof(float));
    MatrixOut = (float*)calloc(size, sizeof(float));

    if (!FilesavingTemp || !FilesavingPower || !MatrixOut) {
        fatal("Failed to allocate host pageable memory for input/output files.");
    }

    readinput(FilesavingTemp, grid_rows_val, grid_cols_val, const_cast<char*>(tfile_name));
    readinput(FilesavingPower, grid_rows_val, grid_cols_val, const_cast<char*>(pfile_name));

    // ALLOCATE HOST-MAPPED MEMORY (ZERO-COPY)
    CUDA_CHECK(cudaHostAlloc((void**)&MatrixTemp[0], sizeof(float) * size, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc((void**)&MatrixTemp[1], sizeof(float) * size, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc((void**)&MatrixPower, sizeof(float) * size, cudaHostAllocMapped));

    // GET DEVICE POINTERS FOR THE HOST-MAPPED MEMORY
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&d_MatrixTemp[0], MatrixTemp[0], 0));
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&d_MatrixTemp[1], MatrixTemp[1], 0));
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&d_MatrixPower, MatrixPower, 0));

    // COPY DATA FROM PAGEABLE HOST MEMORY TO HOST-MAPPED MEMORY (CPU-to-CPU memcpy)
    // This makes the data immediately available to the GPU via the device pointers.
    memcpy(MatrixTemp[0], FilesavingTemp, sizeof(float) * size);
    memcpy(MatrixTemp[1], FilesavingTemp, sizeof(float) * size); // Initialize second buffer too
    memcpy(MatrixPower, FilesavingPower, sizeof(float) * size);

    printf("Start computing the transient temperature\n");

    program_start_time = std::chrono::steady_clock::now();
    controltime = std::chrono::steady_clock::now();
    initializeCudaEvents();
    std::future<void> asyncTask;
    
    //init(); // Uncomment if you need parent PID for signals
    //kill(pids, SIGNAL_TYPE); // Uncomment if you need to send initial signal
    sharedData->newperiods[SHARED_INDEX_FOR_PERIOD] = period_arg; // Use period_arg

    while (keepRunning) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(now - program_start_time).count() / 1000.0;
        outputfile << "Task launched at: " << elapsed_time << " sec" << std::endl << std::endl;

        if (elapsed_time > termination_arg) { // Use termination_arg
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
            outputfile << "Average response time: " << avg_resp_time << " ms" << std::endl;
            rtj1 << avg_resp_time << std::endl;
            responsetime.clear();
            controlperiod.push_back(avg_resp_time); // Use global controlperiod
        }

        asyncTask = std::async(std::launch::async, [&]() {
            // Pass the device pointers to compute_tran_temp
            compute_tran_temp(d_MatrixPower, d_MatrixTemp, grid_cols_val, grid_rows_val, total_iterations, 1000, blockCols, blockRows, borderCols, borderRows);
        });

        period1 << sharedData->newperiods[SHARED_INDEX_FOR_PERIOD] << std::endl;
        
        std::chrono::milliseconds sleep_duration(static_cast<long long>(sharedData->newperiods[SHARED_INDEX_FOR_PERIOD] * 1000.0));
        std::this_thread::sleep_for(sleep_duration);
    }
    
    if (asyncTask.valid()) {
        asyncTask.get(); // Ensure the last asynchronous task completes
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    free(FilesavingPower);
    free(FilesavingTemp);
    free(MatrixOut);
    // FREE HOST-MAPPED MEMORY USING cudaFreeHost
    CUDA_CHECK(cudaFreeHost(MatrixTemp[0]));
    CUDA_CHECK(cudaFreeHost(MatrixTemp[1]));
    CUDA_CHECK(cudaFreeHost(MatrixPower));

    slack1.close();
    period1.close();
    rtj1.close();
    outputfile.close();
    preemptiontime.close();
}

int main(int argc, char** argv) {
    if (argc != 4) {
        printf("Run: %s <setpoint> <period> <termination>\n", argv[0]);
        return -1;
    }

    key_t key = ftok("shmfile", 65);
    int shmid = shmget(key, 1024, 0666 | IPC_CREAT);
    bool createdshm;
    if (shmid > 0) {
        sharedData = (SharedData*)shmat(shmid, nullptr, 0);
        createdshm = true;
    } else {
        shmid = shmget(key, sizeof(SharedData), 0666);
        sharedData = (SharedData*)shmat(shmid, nullptr, 0);
    }

    std::signal(SIGHUP, handleSignal);

    setpoint = atof(argv[1]);
    period = atof(argv[2]);
    termination = atof(argv[3]);

    run(setpoint, period, termination);
    controlperiod.clear();
        if (createdshm == true) {
        shmdt(sharedData);
        shmctl(shmid, IPC_RMID, NULL);
    }
    return 0;
}

