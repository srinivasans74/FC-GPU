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
using namespace std;
int pids;

void init() {
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

// Global pointers for pinned and host memory
float *d_A, *d_B, *d_C;
float *h_A, *h_B, *h_C;
float *dev_d_A, *dev_d_B, *dev_d_C;

int N = 1024;
int jobs = 1;
bool keepRunning = true;
bool signalReceived = false;

void handleSignal(int signal) {
    if (signal == SIGHUP) {
        signalReceived = true;
    }
}

std::vector<float> responsetime;
std::vector<float> executiontime;

float min_rate;
float max_rate;

float calculateaverage(const std::vector<float>& times) {
    if (times.empty()) return 0.0f;
    float sum = 0;
    for (const auto& t : times) sum += t;
    return sum / times.size();
}

std::chrono::time_point<std::chrono::steady_clock> controltime;
std::chrono::time_point<std::chrono::steady_clock> program_start_time;
std::chrono::time_point<std::chrono::steady_clock> preemptionlaunch;

// Global output streams
std::ofstream outputfile;
std::ofstream rtrdeadline;

void kernellaunch(dim3 gridDim, dim3 blockDim, int size) {
    std::ofstream preemptiontime("logs/preemptiont1.txt", std::ios::app);

    auto start = std::chrono::steady_clock::now();
    auto preemption = std::chrono::steady_clock::now();
    auto preemption_start_duration = std::chrono::duration_cast<std::chrono::milliseconds>(preemption - preemptionlaunch).count();
    preemptiontime << "Preemption time = " << preemption_start_duration << " ms" << std::endl;

    float respsum = 0.0f;
    auto cpu_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < jobs; ++i) {
        cudaEvent_t startEvent, stopEvent;
        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);

        cudaEventRecord(startEvent);
        matMulKernel<<<gridDim, blockDim>>>(dev_d_A, dev_d_B, dev_d_C, N);
        cudaEventRecord(stopEvent);
        cudaEventSynchronize(stopEvent);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);

        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);

        respsum += milliseconds;
        outputfile << "Jobtime = \t " << milliseconds << " ms" << std::endl;
    }

    auto cpu_end = std::chrono::high_resolution_clock::now();
    float cpu_time = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();

    responsetime.push_back(cpu_time / jobs + preemption_start_duration);
    executiontime.push_back(cpu_time / jobs);

    if (responsetime.size() > 100) responsetime.erase(responsetime.begin());
    if (executiontime.size() > 100) executiontime.erase(executiontime.begin());

    outputfile << "responsetime= \t" << respsum / jobs + preemption_start_duration / 1000 << " ms" << std::endl;
    preemptiontime.close();
}

float calculatePercentage(const std::vector<float>& times, float deadline_ms) {
    int count = 0;
    for (float t : times) {
        if (t > deadline_ms) count++;
    }
    return times.empty() ? 0.0f : (float)count / times.size() * 100;
}

char stepsize;

void runService(float setpoint, float period, float termination) {
    int size = N * N;

    // Allocate pinned host memory
    if (cudaMallocHost((void**)&d_A, size * sizeof(float)) != cudaSuccess ||
        cudaMallocHost((void**)&d_B, size * sizeof(float)) != cudaSuccess ||
        cudaMallocHost((void**)&d_C, size * sizeof(float)) != cudaSuccess) {
        std::cerr << "Failed to allocate pinned memory" << std::endl;
        return;
    }

    // Map to device pointers
    cudaHostGetDevicePointer(&dev_d_A, d_A, 0);
    cudaHostGetDevicePointer(&dev_d_B, d_B, 0);
    cudaHostGetDevicePointer(&dev_d_C, d_C, 0);

    h_A = (float*)malloc(size * sizeof(float));
    h_B = (float*)malloc(size * sizeof(float));
    h_C = (float*)malloc(size * sizeof(float));

    for (int i = 0; i < size; ++i) {
        h_A[i] = static_cast<float>(i % 100);
        h_B[i] = static_cast<float>(i % 100);
    }

    memcpy(d_A, h_A, size * sizeof(float));
    memcpy(d_B, h_B, size * sizeof(float));

    program_start_time = std::chrono::steady_clock::now();
    controltime = program_start_time;

    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    std::string s_slack   = "logs/s1" + std::string(1, stepsize) + ".txt";
    std::string s_period  = "logs/p1" + std::string(1, stepsize) + ".txt";
    std::string s_rtj     = "logs/rtj1" + std::string(1, stepsize) + ".txt";
    std::string s_et      = "logs/et1" + std::string(1, stepsize) + ".txt";

    std::ofstream slack1(s_slack);
    std::ofstream period1(s_period);
    std::ofstream rtj1(s_rtj);
    std::ofstream executiontimef(s_et);

    while (keepRunning) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - program_start_time).count() / 1000.0;
        outputfile << "Task launched at: " << elapsed << " sec\n\n";

        if (elapsed > termination) {
            std::cout << "Termination time reached. Exiting.\n";
            break;
        }

        if (std::chrono::duration<double>(now - controltime).count() >= 4.0) {
            float rt = calculateaverage(responsetime);
            float et = calculateaverage(executiontime);
            float rtr = calculatePercentage(responsetime, period * 1000);

            rtrdeadline << rtr << endl;
            executiontimef << et << endl;
            rtj1 << rt << endl;
            period1 << period << endl;
            slack1 << (rt / (period * 1000)) << endl;

            if (rt / (period * 1000) < setpoint) {
                period -= (stepsize == 'a') ? 0.045 : (stepsize == 'b') ? 0.095 : 0.145;
            } else {
                period += (stepsize == 'a') ? 0.045 : (stepsize == 'b') ? 0.095 : 0.145;
            }

            responsetime.clear();
            executiontime.clear();
            controltime = std::chrono::steady_clock::now();
        }

        preemptionlaunch = std::chrono::steady_clock::now();
        auto asyncTask = std::async(std::launch::async, [&]() {
            kernellaunch(gridDim, blockDim, size);
        });

        if (asyncTask.valid()) {
            asyncTask.get();
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<long long>(period)));
    }

    cudaFreeHost(d_A);
    cudaFreeHost(d_B);
    cudaFreeHost(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    slack1.close();
    period1.close();
    rtj1.close();
    executiontimef.close();
}

int main(int argc, char *argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <setpoint> <period> <termination> <stepsize (a|b|c)>\n";
        return 1;
    }

    setpoint = atof(argv[1]);
    period = atof(argv[2]);
    termination = atof(argv[3]);
    stepsize = argv[4][0];

    min_rate = period - period * 0.90;
    max_rate = period + period * 4.90;

    std::signal(SIGHUP, handleSignal);

    std::string outLog = "logs/log1" + std::string(1, stepsize) + ".txt";
    std::string rtrLog = "logs/rtrdeadlinet1" + std::string(1, stepsize) + ".txt";
    outputfile.open(outLog);
    rtrdeadline.open(rtrLog);

    std::cout << "Min rate = \t" << min_rate << " Max rate = " << max_rate << "\n";
    std::cout << "Termination = " << termination << "\n";
    std::cout << "Service is running. Press Ctrl+C to exit.\n";

    runService(setpoint, period, termination);

    outputfile.close();
    rtrdeadline.close();

    std::cout << "Service is shutting down.\n";
    return 0;
}