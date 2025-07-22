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

void init()
{
    FILE *file = fopen("logs/mainpid.txt", "r");
    int value;
    fscanf(file, "%d", &value);
    pids = value;
    fclose(file);
    std::cout << "PID PARENT T1= " << pids << endl;
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

// Global pointers for GPU pinned memory and CPU memory
// GPU memory is now allocated using cudaMallocHost (pinned memory)
float *d_A, *d_B, *d_C;
// CPU memory is allocated using malloc
float *h_A, *h_B, *h_C;

int N = 1500; // Example matrix size
int jobs = 2;
bool keepRunning = true;
bool signalReceived = false;
void handleSignal(int signal) {
    if (signal == SIGHUP) {
        bool signalReceived = true;
    }
}

std::vector<float> responsetime;
std::vector<float> executiontime;
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
#define MAKE_LOG_PATH(prefix) ("logs/" prefix LOG_SUFFIX ".txt")
// Global log file streams (initial names; they will be reinitialized in main)
std::string outputFilePath = "logs/log1.txt";
std::string rtrdeadlinePath  = "logs/rtrdeadlinet1.txt";
std::ofstream outputfile(outputFilePath);
std::ofstream rtrdeadline(rtrdeadlinePath);


void kernellaunch(dim3 gridDim, dim3 blockDim, int size) {
    std::ofstream preemptiontime("logs/preemptiont1.txt", std::ios::app);

    auto start = std::chrono::steady_clock::now();
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

        // Perform matrix multiplication using cuBLAS
        matMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);

        // Record the stop event
        cudaEventRecord(stopEvent);
        cudaEventSynchronize(stopEvent);

        // Calculate the elapsed time between start and stop events
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);

        // Destroy the events
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);

        // Accumulate the elapsed time
        respsum += milliseconds;

        // Output the duration
        outputfile << "Jobtime = \t " << milliseconds << " ms" << std::endl;
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    float cpu_time = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();
    responsetime.push_back(cpu_time/jobs + preemption_start_duration/(1)); 
    executiontime.push_back(cpu_time/jobs);
    outputfile << "responsetime= \t" << respsum/jobs + preemption_start_duration/(1) << " ms" << std::endl;
    preemptiontime.close();
}

float calculatePercentage(const vector<float>& responsetime, float x) {
    int count = 0;
    int total = responsetime.size();
    
    // Iterate through the responsetime vector
    for (float value : responsetime) {
        if (value / x > 1) {
            count++;
        }
    }

    // Calculate the percentage
    float percentage = static_cast<float>(count) / total * 100;
    return percentage;
}

// Global variable for stepsize input ('a', 'b', or 'c')
char stepsize;

void runService(float setpoint, float period, float termination) {
    int size = N * N;
    cudaSetDeviceFlags(cudaDeviceMapHost);         

    // --- Allocate GPU pinned memory using cudaMallocHost ---
    //cudaMallocHost((void **)&d_A, size * sizeof(float));
    //cudaMallocHost((void **)&d_B, size * sizeof(float));
    //cudaMallocHost((void **)&d_C, size * sizeof(float));
    cudaHostAlloc((void **)&h_A, size * sizeof(float), cudaHostAllocMapped);
    cudaHostAlloc((void **)&h_B, size * sizeof(float), cudaHostAllocMapped);
    cudaHostAlloc((void **)&h_C, size * sizeof(float), cudaHostAllocMapped);

    // Allocate CPU memory using malloc
   // h_A = (float *)malloc(size * sizeof(float));
   // h_B = (float *)malloc(size * sizeof(float));
   // h_C = (float *)malloc(size * sizeof(float));

    // Initialize matrices on the host
    for (int i = 0; i < size; ++i) {
        h_A[i] = static_cast<float>(i % 100); // Example initialization
        h_B[i] = static_cast<float>(i % 100); // Example initialization
    }


    // map those host pointers into device space
    cudaHostGetDevicePointer((void **)&d_A, h_A, 0);
    cudaHostGetDevicePointer((void **)&d_B, h_B, 0);
    cudaHostGetDevicePointer((void **)&d_C, h_C, 0);
    
    // Reduce redundant transfer by performing a single copy from host to GPU pinned memory
    //cudaMemcpy(d_A, h_A, size * sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_B, h_B, size * sizeof(float), cudaMemcpyHostToDevice);

    auto program_start_time = std::chrono::steady_clock::now();
    auto controltime = std::chrono::steady_clock::now();

    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
  
    // Reinitialize local log file names to include the stepsize letter
    std::string s_slack  = "logs/s1"  + std::string(1, stepsize) + ".txt";
    std::string s_period = "logs/p1"  + std::string(1, stepsize) + ".txt";
    std::string s_rtj    = "logs/rtj1" + std::string(1, stepsize) + ".txt";
   std::string executiontimea     = "logs/et1" + std::string(1, stepsize) + ".txt";

    std::ofstream slack1(s_slack);
    std::ofstream period1(s_period);
    std::ofstream rtj1(s_rtj);
    std::ofstream executiontimef(executiontimea);

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

       
            
           if(controlperiodaa.count()>=4)
        {
          float rt=calculateaverage(responsetime);
          float et= calculateaverage(executiontime);
          float rtr=calculatePercentage(responsetime,period*1000);
          rtrdeadline<<rtr<<endl;
            responsetime.clear();   
            executiontime.clear();   
            executiontimef<<et<<endl;
            rtj1<<rt<<endl;
          float error= setpoint- (rt/(period*1000));
           period1<<period<<endl;
          slack1<<(rt/(period*1000))<<endl;
          period=period-0.25*error;

        controltime = std::chrono::steady_clock::now();
        }

        preemptionlaunch = std::chrono::steady_clock::now();
        
        std::future<void> asyncTask;
        if (asyncTask.valid()) {
            asyncTask.get();
        }
        
        asyncTask = std::async(std::launch::async, [&]() {
            kernellaunch(gridDim, blockDim, size);
        });

        std::chrono::milliseconds sleep_duration(static_cast<long long>(period));
        std::this_thread::sleep_for(sleep_duration);
    }

    // Wait for any pending async task
    // Free GPU pinned memory and CPU memory before exiting
    cudaFreeHost(h_A);         // releases the page-locked, mapped buffers
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    //free(h_A);
    //free(h_B);
    //free(h_C);
    slack1.close();
    period1.close();
    rtj1.close();
    executiontimef.close();

}

int main(int argc, char *argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <setpoint> <period> <termination> <stepsize (a, b, or c)>" << std::endl;
        return 1;
    }

    setpoint = atof(argv[1]);
    period = atof(argv[2]);
    termination = atof(argv[3]);
    stepsize = argv[4][0];  // New input parameter for stepsize

    min_rate = period - period * 0.90;
    max_rate = period + period * 4.90;
    std::signal(SIGHUP, handleSignal);

    // Reinitialize global log file names to include stepsize letter
    std::string new_outputFilePath  = "logs/log1" + std::string(1, stepsize) + ".txt";
    std::string new_rtrdeadlinePath = "logs/rtrdeadlinet1" + std::string(1, stepsize) + ".txt";
    
    outputfile.close();
    rtrdeadline.close();
    outputfile.open(new_outputFilePath);
    rtrdeadline.open(new_rtrdeadlinePath);

    std::cout << "Min rate = \t" << min_rate << "Max rate = " << max_rate << std::endl;
    std::cout << "Terminaition " << termination << std::endl;
    std::cout << "Service is running. Press Ctrl+C to exit." << std::endl;
    runService(setpoint, period, termination);
    outputfile.close();
    std::cout << "Service is shutting down." << std::endl;
    
    return 0;
}