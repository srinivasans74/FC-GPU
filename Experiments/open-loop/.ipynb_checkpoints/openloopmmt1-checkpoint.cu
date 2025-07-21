
#include <cuda_runtime.h>
#include <iostream>
#include <csignal>
#include <thread>
#include <chrono>
#include <future>
#include <vector>
#include <ctime>
#include <fstream>

float period;
float setpoint;
float termination;
int pids;

using namespace std;

void init() {
    FILE *file = fopen("logs/mainpid.txt", "r");
    fscanf(file, "%d", &pids);
    fclose(file);
    cout << "PID PARENT T1= " << pids << endl;
}

__global__ void matMulKernel(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float val = 0.0f;
        for (int k = 0; k < N; ++k) val += A[row * N + k] * B[k * N + col];
        C[row * N + col] = val;
    }
}

float *d_A, *d_B, *d_C;
float *dev_d_A, *dev_d_B, *dev_d_C;
float *h_A, *h_B, *h_C;
int N = 512;
int jobs = 1;

std::vector<float> responsetime;
std::vector<float> executiontime;

std::chrono::steady_clock::time_point controltime;
std::chrono::steady_clock::time_point preemptionlaunch;

ofstream outputfile("logs/log1.txt");
ofstream executiontime1("logs/et1.txt");
ofstream rtrdeadline("logs/rtrdeadlinet1.txt");

float calculateaverage(const vector<float>& vec) {
    float sum = 0;
    for (auto& val : vec) sum += val;
    return vec.empty() ? 0 : sum / vec.size();
}

float calculatePercentage(const vector<float>& vec, float x) {
    int count = 0;
    for (float val : vec) if (val / x > 1) count++;
    return vec.empty() ? 0 : (float)count / vec.size() * 100;
}

void kernellaunch(dim3 gridDim, dim3 blockDim, int size) {
    ofstream preemptiontime("logs/preemptiont1.txt", ios::app);
    auto preemption = chrono::steady_clock::now();
    auto delay = chrono::duration_cast<chrono::milliseconds>(preemption - preemptionlaunch).count();
    preemptiontime << "Preemption time = " << delay << " ms\n";

    float total = 0;
    for (int i = 0; i < jobs; ++i) {
        cudaEvent_t st, sp;
        cudaEventCreate(&st);
        cudaEventCreate(&sp);
        cudaEventRecord(st);
        matMulKernel<<<gridDim, blockDim>>>(dev_d_A, dev_d_B, dev_d_C, N);
        cudaEventRecord(sp);
        cudaEventSynchronize(sp);
        float ms;
        cudaEventElapsedTime(&ms, st, sp);
        cudaEventDestroy(st);
        cudaEventDestroy(sp);
        total += ms;
        outputfile << "Jobtime =\t " << ms << " ms\n";
    }
    float et = total / jobs;
    executiontime.push_back(et);
    responsetime.push_back(et + delay);
    outputfile << "responsetime =\t " << et + delay << " ms\n";
    preemptiontime.close();
}

void runService(float setpoint, float period, float termination) {
    int size = N * N;
    cudaSetDeviceFlags(cudaDeviceMapHost);
    h_A = (float*)malloc(size * sizeof(float));
    h_B = (float*)malloc(size * sizeof(float));
    h_C = (float*)malloc(size * sizeof(float));
    for (int i = 0; i < size; ++i) h_A[i] = h_B[i] = i % 100;

    cudaMallocHost(&d_A, size * sizeof(float));
    cudaMallocHost(&d_B, size * sizeof(float));
    cudaMallocHost(&d_C, size * sizeof(float));
    cudaHostGetDevicePointer(&dev_d_A, d_A, 0);
    cudaHostGetDevicePointer(&dev_d_B, d_B, 0);
    cudaHostGetDevicePointer(&dev_d_C, d_C, 0);
    memcpy(d_A, h_A, size * sizeof(float));
    memcpy(d_B, h_B, size * sizeof(float));

    controltime = chrono::steady_clock::now();
    auto start_time = chrono::steady_clock::now(); // Fixed start time

    dim3 blockDim(16, 16);
    dim3 gridDim((N + 15) / 16, (N + 15) / 16);

    future<void> asyncTask;
    ofstream slack1("logs/s1.txt");
    ofstream period1("logs/p1.txt");
    ofstream rtj1("logs/rtj1.txt");

    bool switched = false;

    timespec next_release;
    clock_gettime(CLOCK_MONOTONIC, &next_release);
    float elapsed = 0.0;

    while (elapsed < termination) {
        auto start = chrono::steady_clock::now();
        preemptionlaunch = chrono::steady_clock::now();

        if (asyncTask.valid()) asyncTask.get();
        asyncTask = async(launch::async, [&]() {
            kernellaunch(gridDim, blockDim, size);
        });

        if (!switched && elapsed >= termination / 2) {
            if (asyncTask.valid()) asyncTask.get();
            switched = true;
            cudaFreeHost(d_A); cudaFreeHost(d_B); cudaFreeHost(d_C);
            free(h_A); free(h_B); free(h_C);
            std::cout<<"Switched\n";
            N *= 2; size = N * N;
            cudaMallocHost(&d_A, size * sizeof(float));
            cudaMallocHost(&d_B, size * sizeof(float));
            cudaMallocHost(&d_C, size * sizeof(float));
            cudaHostGetDevicePointer(&dev_d_A, d_A, 0);
            cudaHostGetDevicePointer(&dev_d_B, d_B, 0);
            cudaHostGetDevicePointer(&dev_d_C, d_C, 0);
            h_A = (float*)malloc(size * sizeof(float));
            h_B = (float*)malloc(size * sizeof(float));
            h_C = (float*)malloc(size * sizeof(float));
            for (int i = 0; i < size; ++i) h_A[i] = h_B[i] = i % 100;
            memcpy(d_A, h_A, size * sizeof(float));
            memcpy(d_B, h_B, size * sizeof(float));
            controltime = chrono::steady_clock::now();
        }

        auto ctl = chrono::steady_clock::now();
        if (chrono::duration<double>(ctl - controltime).count() >= 4.0) {
            float rt = calculateaverage(responsetime);
            float et = calculateaverage(executiontime);
            float rtr = calculatePercentage(responsetime, period * 1000);
            rtrdeadline << rtr << endl;
            executiontime1 << et << endl;
            responsetime.clear();
            executiontime.clear();
            float slack = rt / (period * 1);
            period1 << period << endl;
            slack1 << slack << endl;
            controltime = ctl;
        }

        next_release.tv_nsec += (long)(period * 1e6);
        while (next_release.tv_nsec >= 1e9) {
            next_release.tv_nsec -= 1e9;
            next_release.tv_sec += 1;
        }
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next_release, NULL);

        elapsed = chrono::duration_cast<chrono::milliseconds>(
            chrono::steady_clock::now() - start_time
        ).count() / 1000.0;
    }

    if (asyncTask.valid()) asyncTask.get();
    cudaFreeHost(d_A); cudaFreeHost(d_B); cudaFreeHost(d_C);
    free(h_A); free(h_B); free(h_C);
    slack1.close(); period1.close(); rtj1.close();
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        cerr << "Usage: " << argv[0] << " <setpoint> <period(ms)> <termination(sec)>" << endl;
        return 1;
    }
    setpoint = atof(argv[1]);
    period = atof(argv[2]);
    termination = atof(argv[3]);
    cout << "Service running with nanosleep. Ctrl+C to exit." << endl;
    runService(setpoint, period, termination);
    outputfile.close();
    cout << "Service shut down." << endl;
    return 0;
}
