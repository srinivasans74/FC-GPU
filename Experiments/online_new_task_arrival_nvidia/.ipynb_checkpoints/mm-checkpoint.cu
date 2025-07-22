#include <cuda_runtime.h>
#include <iostream>
#include <csignal>
#include <thread>
#include <chrono>
#include <future>
#include <vector>
#include"shared_data.h"

float period;
float setpoint;
float termination;
using namespace std;
int pids;


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



void init()
{
    FILE *file = fopen("logs/mainpid.txt", "r");
    int value;
    fscanf(file, "%d", &value);
    pids=value;
    fclose(file);
    std::cout<<"PID PARENT T1= "<<pids<<endl;

}




// Kernel for matrix multiplication
__global__ void matMulKernel(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float value = 0.0f;
        for (int k = 0; k < N; ++k) {
            value += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}


float *h_A, *h_B, *h_C;        // host views (pinned, mapped)
float *d_A, *d_B, *d_C;        // device aliases from cudaHostGetDevicePointer
int N = 512; // Example matrix size
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
    float avg_resp_time=0;
    for (const auto& time : resp_time) {
        avg_resp_time += time;
    }
    avg_resp_time /= resp_time.size();
    return avg_resp_time;
}
std::chrono::time_point<std::chrono::steady_clock> controltime;
std::chrono::time_point<std::chrono::steady_clock> program_start_time;
std::chrono::time_point<std::chrono::steady_clock> preemptionlaunch;


//cudaEvent_t start, stop;
//cudaEventCreate(&start);
//cudaEventCreate(&stop);

cudaEvent_t start, stop;
struct timespec starta, enda;

// Initialize CUDA events for timing
void initializeCudaEvents() 
{
   cudaEventCreate(&start);
    cudaEventCreate(&stop);
}



void kernellaunch(dim3 gridDim, dim3 blockDim, int size) {
    auto start_chrono = std::chrono::steady_clock::now();
    auto preemption = std::chrono::steady_clock::now();
    auto preemption_start_duration = std::chrono::duration_cast<std::chrono::milliseconds>(preemption - preemptionlaunch).count();
    preemptiontime << "Preemption time = " << preemption_start_duration / 1000.0 << " ms" << std::endl;
    clock_gettime(CLOCK_MONOTONIC_RAW, &enda);
    preemption_start_duration = (enda.tv_sec - starta.tv_sec) * 1e3 + (enda.tv_nsec - starta.tv_nsec) / 1e6; // in milliseconds
    float kernel_elapsed_time = 0.0f;
    float respsum = 0.0f;

    for (int i = 0; i < jobs; ++i) {
        // Record the start event
        cudaEventRecord(start);
        matMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&kernel_elapsed_time, start, stop);
        respsum += kernel_elapsed_time;

        outputfile << "Job time = \t" << kernel_elapsed_time << " ms" << std::endl;
    }
    responsetime.push_back((respsum / jobs) + preemption_start_duration / 1.0);
    sharedData->values[SHARED_MEM_INDEX] = (respsum / jobs) + preemption_start_duration / 1.0;
}

void runService(float setpoint, float period, float termination) {
    int size = N * N;

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

    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    auto program_start_time = std::chrono::steady_clock::now();
    auto controltime = std::chrono::steady_clock::now();

    std::future<void> asyncTask;
    std::vector<float> controlperiod;

    //init();
    //kill(pids, SIGNAL_TYPE);
    sharedData->newperiods[SHARED_MEM_INDEX2] = period;
    initializeCudaEvents();
    while (keepRunning) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(now - program_start_time).count() / 1000.0;
        outputfile << "Task launched at: " << elapsed_time << " sec" << std::endl << std::endl;

        if (elapsed_time > termination) {
            std::cout << "Termination time reached. Exiting." << std::endl;
            break;
        }

        preemptionlaunch = std::chrono::steady_clock::now();
        clock_gettime(CLOCK_MONOTONIC_RAW, &starta);

        if (asyncTask.valid()) {
            asyncTask.get();
        }

        // Launch the asynchronous task for kernel execution
        asyncTask = std::async(std::launch::async, [&]() {
            kernellaunch(gridDim, blockDim, size);
            float avg_resp_time = calculateaverage(responsetime);
            outputfile << "Average response time: " << avg_resp_time << " ms" << std::endl;
            rtj1 << avg_resp_time << std::endl;
            responsetime.clear();
            controlperiod.push_back(avg_resp_time);
        });

        period1 << period << std::endl;

        // Sleep for the specified period
        std::chrono::milliseconds sleep_duration(static_cast<long long>(sharedData->newperiods[SHARED_MEM_INDEX2] * 1000.0));
        std::this_thread::sleep_for(sleep_duration);
    }

    if (asyncTask.valid()) {
        asyncTask.get();
    }

    // Destroy the CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);          // releases both host & device views in one call

    // Close files and clean up
    slack1.close();
    period1.close();
    rtj1.close();
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
