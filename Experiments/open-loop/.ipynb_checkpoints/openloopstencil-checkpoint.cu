#include <cuda_runtime.h>
#include <iostream>
#include <csignal>
#include <thread>
#include <chrono>
#include <future>
#include <vector>
#include <iomanip>
#include <ctime>
#include <fstream>

using namespace std;
using steady_clock = std::chrono::steady_clock;

float setpoint, period_ms, termination_ms;
const int JOBS = 1;
bool workload_changed = false;

#define BSIZE 16
#define XTILE 20

const int size = 512;
int repeat = 375;
int nx = size, ny = size, nz = size;
int vol = nx * ny * nz;

typedef double Real;
Real *h_Vm, *h_dVm, *h_sigma;   // host, page-locked
Real *d_Vm, *d_dVm, *d_sigma;   // GPU aliases obtained with cudaHostGetDevicePointer
std::vector<float> responsetime;
std::vector<float> executiontime;

steady_clock::time_point controltime;
steady_clock::time_point preemptionlaunch;

ofstream outputfile("logs/log2.txt");
ofstream executiontime1("logs/et2.txt");
ofstream rtrdeadline("logs/rtrdeadlinet2.txt");
ofstream slack1("logs/s2.txt");
ofstream period1("logs/p2.txt");
ofstream rtj1("logs/rtj2.txt");
ofstream preemptiontime("logs/preemptiont1.txt", ios::app);

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

    int pii = 0, cii = 1, nii = 2, tii;

    d_psi = &(d_psi[XTILE * blockIdx.x + nx * ((BSIZE - 2) * blockIdx.y + ny * (BSIZE - 2) * blockIdx.z)]);
    d_npsi = &(d_npsi[XTILE * blockIdx.x + nx * ((BSIZE - 2) * blockIdx.y + ny * (BSIZE - 2) * blockIdx.z)]);

    if (tjj >= BSIZE || tkk >= BSIZE) return;

    sm_psi[cii][tjj][tkk] = d_psi[tjj * nz + tkk];
    sm_psi[nii][tjj][tkk] = d_psi[nx * ny + tjj * nz + tkk];

    __syncthreads();

    for (int ii = 1; ii < XTILE; ++ii) {
        sm_psi[nii][tjj][tkk] = d_psi[(ii + 1) * ny * nz + tjj * nz + tkk];
        __syncthreads();
        __syncthreads();
        tii = pii; pii = cii; cii = nii; nii = tii;
    }
}

void allocateAndInitMemory()
{
    // 1. page-locked, mappable allocations
    cudaSetDeviceFlags(cudaDeviceMapHost);           
    cudaHostAlloc(&h_Vm,    sizeof(Real) * vol,     cudaHostAllocMapped);
    cudaHostAlloc(&h_sigma, sizeof(Real) * vol * 9, cudaHostAllocMapped);
    cudaHostAlloc(&h_dVm,   sizeof(Real) * vol,     cudaHostAllocMapped);

    // 2. get the matching device pointers
    cudaHostGetDevicePointer((void**)&d_Vm,    h_Vm,    0);
    cudaHostGetDevicePointer((void**)&d_sigma, h_sigma, 0);
    cudaHostGetDevicePointer((void**)&d_dVm,   h_dVm,   0);

    // 3. initialise directly on host
    for (int i = 0; i < vol;      ++i) h_Vm[i]    = i % 19;
    for (int i = 0; i < vol * 9;  ++i) h_sigma[i] = i % 19;
    memset(h_dVm, 0, sizeof(Real) * vol);          // still host memory
}

void freeMemory()
{
    cudaFreeHost(h_Vm);
    cudaFreeHost(h_sigma);
    cudaFreeHost(h_dVm);
}

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

void kernellaunch(dim3 gridDim, dim3 blockDim) {
    auto pre = steady_clock::now();
    float pre_ms = std::chrono::duration<float, std::milli>(pre - preemptionlaunch).count();
    preemptiontime << "Preemption time = " << pre_ms << " ms\n";

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
        outputfile << "Jobtime =\t " << el << " ms\n";
    }

    executiontime.push_back(total_gpu / JOBS);
    responsetime.push_back(pre_ms + total_gpu / JOBS);
    outputfile << "responsetime =\t " << pre_ms + total_gpu / JOBS << " ms\n";
}

void runPeriodicService() {
    allocateAndInitMemory();
    int bdimx = (nx - 2) / XTILE + ((nx - 2) % XTILE != 0);
    int bdimy = (ny - 2) / (BSIZE - 2) + ((ny - 2) % (BSIZE - 2) != 0);
    int bdimz = (nz - 2) / (BSIZE - 2) + ((nz - 2) % (BSIZE - 2) != 0);

    dim3 gridDim(bdimx, bdimy, bdimz);
    dim3 blockDim(BSIZE, BSIZE, 1);

    controltime = chrono::steady_clock::now();
    auto start_time = chrono::steady_clock::now();  // Fixed reference point

    timespec next_release;
    clock_gettime(CLOCK_MONOTONIC, &next_release);
    float elapsed = 0.0;
    const float change_trigger = termination_ms / 2.0f;

    while (elapsed < termination_ms) {
        preemptionlaunch = chrono::steady_clock::now();

        future<void> fut = async(launch::async, [&]() {
            kernellaunch(gridDim, blockDim);
        });
        fut.get();

        if (!workload_changed && elapsed >= change_trigger) {
            std::cout << "[WORKLOAD CHANGE @ " << std::fixed << std::setprecision(2) << elapsed / 1000.0f << " s]\n";
            std::cout << "WORKLOAD CHANGE: Repeat count increased.\n";
            repeat = 500;
            workload_changed = true;
        }

        auto ctl = chrono::steady_clock::now();
        if (chrono::duration<double>(ctl - controltime).count() >= 4.0) {
            float rt = calculateaverage(responsetime);
            float et = calculateaverage(executiontime);
            float rtr = calculatePercentage(responsetime, period_ms);
            rtrdeadline << rtr << endl;
            executiontime1 << et << endl;
            responsetime.clear();
            executiontime.clear();
            float slack = rt / period_ms;
            period1 << period_ms << endl;
            slack1 << slack << endl;
            controltime = ctl;
        }

        next_release.tv_nsec += (long)(period_ms * 1e6);
        while (next_release.tv_nsec >= 1e9) {
            next_release.tv_nsec -= 1e9;
            next_release.tv_sec += 1;
        }
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next_release, NULL);

        elapsed = std::chrono::duration<float, std::milli>(chrono::steady_clock::now() - start_time).count();  // ✅ FIXED
    }

    freeMemory();
    slack1.close(); period1.close(); rtj1.close(); outputfile.close(); executiontime1.close(); rtrdeadline.close(); preemptiontime.close();
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        cerr << "Usage: " << argv[0] << " <setpoint> <period(ms)> <termination(s)>" << endl;
        return 1;
    }

    setpoint = atof(argv[1]);
    period_ms = atof(argv[2]);
    termination_ms = atof(argv[3]) * 1000.0f;

    cout << "[T2] Real-Time Stencil Task Started.\n";
    runPeriodicService();
    cout << "[T2] Service shut down.\n";

    return 0;
}