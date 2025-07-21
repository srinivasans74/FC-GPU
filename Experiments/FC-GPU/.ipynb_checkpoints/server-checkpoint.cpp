#include "shared_data.h"
#include <csignal>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <unistd.h>
#include <sys/ipc.h>
#include <sys/shm.h>

using namespace std;

bool signalReceived = false;
bool signalReceived2 = false;

int count = 10;
std::string t1, t2;

std::unordered_map<std::string, int> mappingTable = {
    {"t1", SIGUSR1},
    {"t2", SIGUSR2},
};

void signalHandler1(int signum) {
    if (signum == mappingTable[t1]) {
        signalReceived = true;
    } else if (signum == mappingTable[t2]) {
        signalReceived2 = true;
    }
    count++;
}

void init() {
    int pid = getpid();
    std::cout << pid << endl;
    FILE* file1 = fopen("logs/mainpid.txt", "w");
    fprintf(file1, "%d\n", pid);
    fclose(file1);

    std::ifstream file("logs/task_values.txt");
    std::string line;
    int currentLine = 1;

    std::unordered_map<std::string, std::pair<int, int>> mappings = {
        {"t1", {0, 2}},
        {"t2", {1, 3}},
    };

    while (std::getline(file, line)) {
        if (!line.empty() && line[0] == 't') {
            std::string key = line.substr(0, 2);
            auto it = mappings.find(key);
            if (it != mappings.end()) {
                if (currentLine == 1) {
                    sm1 = it->second.first;
                    sm11 = it->second.second;
                } else if (currentLine == 2) {
                    sm2 = it->second.first;
                    sm22 = it->second.second;
                }
                currentLine++;
            }
        }
    }

    file.clear();
    file.seekg(0);

    std::unordered_map<std::string, int> procMapping = {
        {"t1", 0},
        {"t2", 1},
    };

    currentLine = 1;
    while (std::getline(file, line)) {
        if (!line.empty() && line[0] == 't') {
            std::string key = line.substr(0, 2);
            auto it = procMapping.find(key);
            if (it != procMapping.end()) {
                if (currentLine == 1) {
                    proc11 = it->second;
                } else if (currentLine == 2) {
                    proc22 = it->second;
                }
                currentLine++;
            }
        }
    }
    file.close();
}

float calculateDeadlineMiss(const std::vector<float>& responseTimes, float period) {
    int missed = 0;
    for (float rt : responseTimes) {
        if (rt > period) missed++;
    }
    return responseTimes.empty() ? 0.0f : (100.0f * missed / responseTimes.size());
}


int main(int argc, char* argv[]) {
    int tasks = atoi(argv[1]);
    if (tasks != 2) {
        std::cerr << "Only 2-task configuration is supported in this version." << std::endl;
        return 1;
    }

    key_t key = ftok("shmfile", 65);
    int shmid = shmget(key, sizeof(SharedData), 0666);
    SharedData* sharedData = (SharedData*)shmat(shmid, nullptr, 0);

    init();
    float controlperiod = 4.0;

    std::ifstream file("logs/task_values.txt");
    file >> t1 >> t2;
    file.close();

    signal(mappingTable[t1], signalHandler1);
    signal(mappingTable[t2], signalHandler1);

    int pida = atoi(argv[2]);
    int pidb = atoi(argv[3]);
    float setpoint1 = atof(argv[4]);
    float setpoint2 = atof(argv[5]);

    std::ofstream slack1("logs/s11.txt");
    std::ofstream slack2("logs/s22.txt");
    std::ofstream deadlineMiss1("logs/deadlinemisst1.txt");
    std::ofstream deadlineMiss2("logs/deadlinemisst2.txt");
    std::ofstream durationarrra("logs/durationarray.txt");

    std::ofstream int1("logs/int1.txt");
    std::ofstream int2("logs/int2.txt");

    // NEW: save current shared periods
    std::ofstream rt1("logs/rt1.txt");
    std::ofstream rt2("logs/rt2.txt");

    average avg;
    sharedData->newperiods[proc11] = 0;
    sharedData->newperiods[proc22] = 0;

    // while (count < 2) {
    //     usleep(90000);
    // }

    // kill(pida, SIGHUP);
    // kill(pidb, SIGHUP);

    float a11 = 0.001700 *1;
    float a12 = 0.0001968 *1;
    float a21 = 0.0001968 *1;
    float a22 = 0.001700*1;
    std::cout<<"\n \n Gain\n ***************************\n \n";
    std::cout <<a11 << " " << a12 << std::endl;
    std::cout <<a21 << " " << a22 << std::endl;
    std::cout<<"\n \n \n ***************************\n \n";
    usleep(10000000);

    float sx = 0;
    float error1 = 0, error2 = 0;

    while (true) {
        std::cout << "\n****** Control Period: " << sx << " ******\n";

        float prevValue1 = 0, prevValue2 = 0;
        auto start = std::chrono::high_resolution_clock::now();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;

        while (duration.count() < controlperiod) {
            usleep(10000);
            end = std::chrono::high_resolution_clock::now();
            duration = end - start;

            if (sharedData->values[sm1] != prevValue1) {
                int1 << sharedData->values[sm1] << std::endl;
                avg.taskexec1.push_back(sharedData->values[sm1] / (sharedData->newperiods[proc11] * 1));
                avg.taskexec11.push_back(sharedData->values[sm1]);
                prevValue1 = sharedData->values[sm1];
            }
            if (sharedData->values[sm2] != prevValue2) {
                int2 << sharedData->values[sm2] << std::endl;
                avg.taskexec2.push_back(sharedData->values[sm2] / (sharedData->newperiods[proc22] * 1));
                avg.taskexec22.push_back(sharedData->values[sm2]);
                prevValue2 = sharedData->values[sm2];
            }
        }

        int1 << "Control Period\n";
        int2 << "Control Period\n";
  
        float avg_task11 = avg.calculateAverage(avg.taskexec11);
        float avg_task22 = avg.calculateAverage(avg.taskexec22);

        float avg_task1 = avg.calculateAverage(avg.taskexec1);
        float avg_task2 = avg.calculateAverage(avg.taskexec2);

        slack1 << avg_task1 << endl;
        slack2 << avg_task2 << endl;

        deadlineMiss1 << calculateDeadlineMiss(avg.taskexec11, sharedData->newperiods[proc11]) << endl;
        deadlineMiss2 << calculateDeadlineMiss(avg.taskexec22, sharedData->newperiods[proc22]) << endl;

        float sep = setpoint1 - avg_task1;
        float sep2 = setpoint2 - avg_task2;

        float s1 = (sep * a11) + (sep2 * a12);
        float s2 = (sep * a21) + (sep2 * a22);

        float period1 = sharedData->newperiods[proc11];
        float period2 = sharedData->newperiods[proc22];
        
        rt1 << sharedData->newperiods[proc11] << std::endl;
        rt2 << sharedData->newperiods[proc22] << std::endl;
        

        float new1 = period1 / (1 + (period1 * period2 * s1));
        float new2 = period2 / (1 + (period2 * period1 * s2));

        sharedData->newperiods[proc11] = new1;
        sharedData->newperiods[proc22] = new2;

        // NEW: log current periods


        std::cout << "RT 1 = \t" << avg_task11 << "\tRT 2 = \t" << avg_task22 << std::endl;
        std::cout << "Period 1 = \t" << period1*1 << "\tPeriod 2 = \t" << period2*1 << std::endl;
        std::cout << "S1 = \t" << avg_task1 << "\tS2 = \t" << avg_task2 << std::endl;
        std::cout << "Error1 = " << sep << "\tError2 = " << sep2 << std::endl;
        std::cout << "Delta1 = \t" << s1 << "\tDelta2 = \t" << s2 << std::endl;
        std::cout << "New Period 1 = " << new1 << "\tNew Period 2 = " << new2 << std::endl;

        durationarrra << sx << std::endl;

        avg.taskexec1.clear();
        avg.taskexec2.clear();
        avg.taskexec11.clear();
        avg.taskexec22.clear();

        signalReceived = false;
        signalReceived2 = false;
        count = 0;
        sx += 1;
    }

    // Cleanup
    shmdt(sharedData);
    slack1.close(); slack2.close();
    deadlineMiss1.close(); deadlineMiss2.close();
    durationarrra.close();
    int1.close(); int2.close();
    rt1.close(); rt2.close(); // NEW

    return 0;
}