#include "shared_data1.h"
#include <chrono>
#include <thread>
#include <iostream>
#include <fstream>
#include <vector>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <unistd.h>
#include <sys/stat.h>   // mkdir
#include <sys/mman.h>
#include <sched.h>
#include <pthread.h>

using namespace std;
using namespace std::chrono;

void tryEnableRealtimeScheduling() {
    static bool attempted = false;
    if (attempted) return;
    attempted = true;

    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        perror("mlockall");
        cerr << "Warning: mlockall failed. Proceeding without memory lock.\n";
    }

    sched_param param{};
    param.sched_priority = 80;
    if (sched_setscheduler(0, SCHED_FIFO, &param) != 0) {
        perror("sched_setscheduler");
        cerr << "Warning: SCHED_FIFO requires sudo. Using default scheduler.\n";
    }

    cpu_set_t cp;
    CPU_ZERO(&cp);
    CPU_SET(0, &cp);
    if (sched_setaffinity(0, sizeof(cp), &cp) != 0) {
        perror("sched_setaffinity");
        cerr << "Warning: Failed to set CPU affinity. Proceeding without binding.\n";
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <duration_in_seconds> <threshold>\n";
        return 1;
    }

    int duration_seconds = atoi(argv[1]);
    float setpoint = atof(argv[2]);

    // Create logs directory if needed
    mkdir("tdmalogs", 0777);

    auto start_time = steady_clock::now();

    key_t key = ftok("shmfile", 65);
    int shmid = shmget(key, sizeof(SharedData), 0666 | IPC_CREAT);
    if (shmid < 0) {
        perror("shmget");
        return 1;
    }

    SharedData* shm = static_cast<SharedData*>(shmat(shmid, nullptr, 0));
    if (shm == (void*)-1) {
        perror("shmat");
        return 1;
    }

    // tryEnableRealtimeScheduling();  // Uncomment if desired

    // Log files
    ofstream rtr1("tdmalogs/rtr1.txt"), rtr2("tdmalogs/rtr2.txt");
    ofstream rt1("tdmalogs/responetime1.txt"), rt2("tdmalogs/responetime2.txt");
    ofstream cpu1("tdmalogs/cpu_rt1.txt"), cpu2("tdmalogs/cpu_rt2.txt");
    ofstream p1("tdmalogs/period1.txt"), p2("tdmalogs/period2.txt");
    ofstream miss1("tdmalogs/misses1.txt"), miss2("tdmalogs/misses2.txt");

    if (!rtr1 || !rtr2 || !rt1 || !rt2 || !cpu1 || !cpu2 || !p1 || !p2 || !miss1 || !miss2) {
        cerr << "Error opening one or more log files.\n";
        return 1;
    }

    int slices[2] = {10, 10};
    const int CONTROL_MS = 400;
    const int WINDOW_TICKS = 10;
    this_thread::sleep_for(milliseconds(CONTROL_MS));

    float sumRT[2] = {0.f, 0.f};
    float sumCPU[2] = {0.f, 0.f};
    int cntRT[2] = {0, 0};
    int missCount[2] = {0, 0};
    int tick = 0;

    while (steady_clock::now() - start_time < seconds(duration_seconds)) {
        for (int id = 0; id < 2; ++id) {
            float gpuRT = shm->values[id];
            float cpuRT = shm->executiontime[id];
            float period = shm->newperiods[id] > 0 ? shm->newperiods[id] : 1.0f;

            sumRT[id] += gpuRT;
            sumCPU[id] += cpuRT;
            cntRT[id]++;

            if (gpuRT > period)
                missCount[id]++;
        }

        tick++;

        if (tick >= WINDOW_TICKS) {
            for (int id = 0; id < 2; ++id) {
                float avgGPU = cntRT[id] ? sumRT[id] / cntRT[id] : 0.f;
                float avgCPU = cntRT[id] ? sumCPU[id] / cntRT[id] : 0.f;
                float period = shm->newperiods[id] > 0 ? shm->newperiods[id] : 1.0f;
                float avgRTR = period > 0 ? avgGPU / period : 0.f;
                float missRate = cntRT[id] ? (missCount[id] * 100.0f / cntRT[id]) : 0.f;

                shm->slices[id] = slices[id];

                auto current_time = steady_clock::now();
                auto elapsed = duration_cast<seconds>(current_time - start_time).count();

                cout << "[" << elapsed << "s] [CTL] id=" << id
                     << "  avgGPU_RT=" << avgGPU
                     << "  avgCPU_RT=" << avgCPU
                     << "  RTR=" << avgRTR
                     << "  slices=" << slices[id]
                     << "  period=" << period
                     << "  misses=" << missRate << "%\n";

                // Logging
                if (id == 0) {
                    rt1 << avgGPU << '\n'; rt1.flush();
                    cpu1 << avgCPU << '\n'; cpu1.flush();
                    rtr1 << avgRTR << '\n'; rtr1.flush();
                    p1 << period << '\n';  p1.flush();
                    miss1 << missCount[id] << '\n'; miss1.flush();
                } else {
                    rt2 << avgGPU << '\n'; rt2.flush();
                    cpu2 << avgCPU << '\n'; cpu2.flush();
                    rtr2 << avgRTR << '\n'; rtr2.flush();
                    p2 << period << '\n';  p2.flush();
                    miss2 << missCount[id] << '\n'; miss2.flush();
                }

                // Reset for next window
                sumRT[id] = sumCPU[id] = 0.f;
                cntRT[id] = missCount[id] = 0;
            }

            tick = 0;
            cerr << "------------------------------------------------\n";
        }

        this_thread::sleep_for(milliseconds(CONTROL_MS));
    }

    // Cleanup
    shmdt(shm);
    rtr1.close(); rtr2.close();
    rt1.close(); rt2.close();
    cpu1.close(); cpu2.close();
    p1.close(); p2.close();
    miss1.close(); miss2.close();

    return 0;
}