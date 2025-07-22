#include "shared_data.h"
using namespace std;

bool signalReceived = false;
bool signalReceived2 = false;
bool signalReceived3 = false;
bool signalReceived4 = false;

int count = 10;
std::string t1;
std::string t2, t3, t4;
std::unordered_map<std::string, int> mappingTable = {
    {"t1", SIGUSR1},
    {"t2", SIGUSR2},
    {"t3", SIGUSR3},
    {"t4", SIGUSR4}
};


void runPythonScript(const std::string& scriptName, const std::string& args) {
    std::string command = "python3 " + scriptName + " " + args;
    int result = std::system(command.c_str());
    if (result != 0) {
        std::cerr << "Error running Python script" << std::endl;
        // Handle the error if necessary
    }
}


// Signal handler for task 1
void signalHandler1(int signum) {
    if (signum == mappingTable[t1]) {
        signalReceived = true;
    } else if (signum == mappingTable[t2]) {
        signalReceived2 = true;
    }
    count++;
}

void signalHandler2(int signum) {
    if (signum == mappingTable[t1]) {
        signalReceived = true;
    } else if (signum == mappingTable[t2]) {
        signalReceived2 = true;
    } else if (signum == mappingTable[t3]) {
        signalReceived3 = true;
    }

    count++;
}


void signalHandler3(int signum) {
    if (signum == mappingTable[t1]) {
        signalReceived = true;
    } else if (signum == mappingTable[t2]) {
        signalReceived2 = true;
    } else if (signum == mappingTable[t3]) {
        signalReceived3 = true;
    } else if (signum == mappingTable[t4]) {
        signalReceived4 = true;
    }
    count++;
}


void init() {
    int value;
    int pid = getpid();
    std::cout << pid << endl;
    FILE *file1 = fopen("logs/mainpid.txt", "w");
    fprintf(file1, "%d\n", pid);
    fclose(file1);
    std::string line;
    int currentLine = 1;

    std::ifstream file("logs/task_values.txt");
    std::unordered_map<std::string, std::pair<int, int>> mappings = {
        {"t1", {0, 2}},
        {"t2", {1, 3}},
        {"t3", {4, 6}},
        {"t4", {5, 7}}
    };


    while (std::getline(file, line)) {
        if (!line.empty() && line[0] == 't') {
            std::string key = line.substr(0, 2); // Extract the whole key
            auto it = mappings.find(key);
            if (it != mappings.end()) {
                switch (currentLine) {
                    case 1:
                        sm1 = it->second.first;
                        sm11 = it->second.second;
                        std::cout << "Found: " << key << ", sm1: " << sm1 << ", sm11: " << sm11 << std::endl;
                        break;
                    case 2:
                        sm2 = it->second.first;
                        sm22 = it->second.second;
                        std::cout << "Found: " << key << ", sm2: " << sm2 << ", sm22: " << sm22 << std::endl;
                        break;
                    case 3:
                        sm3 = it->second.first;
                        sm33 = it->second.second;
                        std::cout << "Found: " << key << ", sm3: " << sm3 << ", sm33: " << sm33 << std::endl;
                        break;
                    case 4:
                        sm4 = it->second.first;
                        sm44 = it->second.second;
                        std::cout << "Found: " << key << ", sm4: " << sm4 << ", sm44: " << sm44 << std::endl;
                        break;
                    default:
                        break;
                }
                // Increment currentLine and reset to 1 if it exceeds 4
                currentLine = (currentLine % 4) + 1;
            } else {
                std::cout << "Please use t1, t2, t3, or t4 in that order" << std::endl;
            }
        }

    }
    file.clear();
    file.seekg(0);
    std::unordered_map<std::string, std::pair<int, int>> mappingTable = {
        {"t1", {0, 0}},
        {"t2", {1, 1}},
        {"t3", {2, 2}},
        {"t4", {3, 3}}
    };

    currentLine = 1; // Initialize current line to 1

    while (std::getline(file, line)) {
        if (!line.empty() && line[0] == 't') {
            std::string key = line.substr(0, 2); // Extract the whole key
            auto it = mappingTable.find(key);
            if (it != mappingTable.end()) {
                int value1 = it->second.first;
                int value2 = it->second.second;
                // Assign values to proc11, proc22, proc33, proc44 based on the mapping
                switch (currentLine) {
                    case 1:
                        proc11 = value2;
                        break;
                    case 2:
                        proc22 = value2;
                        break;
                    case 3:
                        proc33 = value2;
                        break;
                    case 4:
                        proc44 = value2;
                        break;
                    default:
                        break;
                }
                // Print debug values
                // std::cout << "Current Line: " << currentLine << ", proc1: " << proc1 << ", proc2: " << proc2
                //             << ", proc3: " << proc3 << ", proc4: " << proc4 << std::endl;

                // Increment currentLine and reset to 1 if it exceeds 4
                currentLine = (currentLine % 4) + 1;
            } else {
                //std::cout << "Please use t1, t2, t3, or t4 in that order" << std::endl;
            }
        }
    }
    std::cout << "Found: " << "proc1: " << proc11 << ", proc2: " << proc22 << ", proc3: " << proc33 << ", proc4: " << proc44 << std::endl;


    file.close();


}

// Simple bubble sort implementation to sort the window
void bubbleSort(std::vector<float>& arr) {
    size_t n = arr.size();
    for (size_t i = 0; i < n - 1; ++i) {
        for (size_t j = 0; j < n - i - 1; ++j) {
            if (arr[j] > arr[j + 1]) {
                std::swap(arr[j], arr[j + 1]);
            }
        }
    }
}

double medianFilter(double new_value, std::deque<float>& window, size_t window_size) {
    window.push_back(new_value);
    if (window.size() > window_size) {
        window.pop_front();
    }

    std::vector<float> sorted_window(window.begin(), window.end());
    bubbleSort(sorted_window);  // Use custom bubble sort to sort the window
    return sorted_window[sorted_window.size() / 2];
}

void applyMedianFilter(std::vector<float>& values, size_t windowSize) {
    std::deque<float> window;

    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = medianFilter(values[i], window, windowSize);  // Replace the original value with the filtered value
    }
}

bool jump = false;

std::atomic<bool> neworkload(false);

// Signal handler for SIGINT
void interupt(int signal) {
    if (signal == SIGINT) {
        std::cout << "SIGINT received new task is arrived" << std::endl;
        // Toggle the signalReceived flag
        neworkload = !neworkload; // Toggle the flag
    }
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


int main(int argc, char* argv[]) {

    int tasks = atoi(argv[1]);
    std::cout << "tasks = " << tasks << endl;
    usleep(3000000);  // Sleep for 2 seconds

    key_t key = ftok("shmfile", 65);
    int shmid = shmget(key, sizeof(SharedData), 0666);
    SharedData* sharedData = (SharedData*)shmat(shmid, nullptr, 0);
    init();
    std::signal(SIGINT, interupt);

    std::ofstream slack1("logs/s11.txt");
    std::ofstream slack2("logs/s22.txt");
    std::ofstream slack3("logs/s33.txt");
    std::ofstream slack4("logs/s44.txt");
    std::ofstream cpt1("logs/cpt1.txt");
    std::ofstream cpt2("logs/cpt2.txt");
    std::ofstream cpt3("logs/cpt3.txt");
    std::ofstream cpt4("logs/cpt4.txt");
    std::ofstream durationarrra("logs/durationarray.txt");
    std::ofstream int1("logs/int1.txt");
    std::ofstream int2("logs/int2.txt");
    std::ofstream int3("logs/int3.txt");
    std::ofstream int4("logs/int4.txt");
    std::ofstream setpointt1("logs/cptold1.txt");
    std::ofstream setpointt2("logs/cptold2.txt");
    std::ofstream setpointt3("logs/cptold3.txt");
    std::ofstream setpointt4("logs/cptold4.txt");
    std::ofstream rtrdeadline1("logs/deadlinemisstrtr_1.txt");
    std::ofstream rtrdeadline2("logs/deadlinemisstrtr_2.txt");
    std::ofstream rtrdeadline3("logs/deadlinemisstrtr_3.txt");
    std::ofstream rtrdeadline4("logs/deadlinemisstrtr_4.txt");

    float setpoint1;
    float setpoint2;
    float setpoint3;
    float setpoint4;
    goto start;
start:
    std::ifstream file("logs/task_values.txt");
    std::string task1_str, task2_str, task3_str, task4_str;
    file >> task1_str >> task2_str >> task3_str >> task4_str;
    file.close();
    t1 = task1_str;
    t2 = task2_str;
    t3 = task3_str;
    t4 = task4_str;
    std::cout << t1 << "\t" << t2 << "\t" << t3 << "\t" << t4 << endl;
    std::cout << "No fo tasks =" << tasks << endl;
    std::cout << t1 << "\t" << t2 << "\t" << t3 << "\t" << t4 << endl;
    file.close();
    tasks = 0;
    if (!task1_str.empty()) tasks++;
    if (!task2_str.empty()) tasks++;
    if (!task3_str.empty()) tasks++;
    if (!task4_str.empty()) tasks++;

    std::cout << "Number of tasks = " << tasks << std::endl;

    // The following print is redundant as we already printed the tasks above
    // But we'll keep it as per the original code request
    std::cout << t1 << "\t" << t2 << "\t" << t3 << "\t" << t4 << std::endl;


    if (tasks == 3) {
        signal(mappingTable[task1_str], signalHandler2);
        signal(mappingTable[task2_str], signalHandler2);
        signal(mappingTable[task3_str], signalHandler2);
        int pida = atoi(argv[2]);
        int pidb = atoi(argv[3]);
        int pidc = atoi(argv[4]);
        setpoint1 = atof(argv[5]) * 1;
        setpoint2 = atof(argv[6]) * 1;
        setpoint3 = atof(argv[7]) * 1;
        std::cout << "No of tasks = 3 " << pida << pidb << pidc << setpoint1 << setpoint2
            << setpoint3 << endl;
        double totautil;
        float sep;
        float sep2;
        float sep3;
        float error3;
        float s11 = 0;
        float s22 = 0;
        float error1 = 0;
        float error2 = 0;
        float sx = 0;
        int updatecheck = 0;
        double previous_error1 = 0;
        double previous_error2 = 0;

        float a11;
        float a12;
        float a13;
        float a21;
        float a22;
        float a23;
        float a31;
        float a32;
        float a33;
        float s1;
        float s2;
        float s3;
        float si1 = 0;
        float si2 = 0;
        float si3 = 0;
        float new1, new2, new3;
        new1 = 0;
        new2 = 0;
        new3 = 0;


        //gains
        a11 = 5.4968 * 2 * 8;
        a12 = 3.3968 / 10;
        a13 = 3.3968 / 10;
        a21 = 3.3968 / 10;
        a22 = 7.968 * 2 * 5;
        a23 = 3.3968 / 10;
        a31 = 2.2976 / 10;
        a32 = 2.2976 / 10;
        a33 = 1.6976 * 2 * 25;


        std::cout << "****************************************************\n";
        std::cout << "****************************************************\n";
        std::cout << "****************************************************\n";
        std::cout << "****************************************************\n";
        std::cout << "****************************************************\n";

        average avg;
        std::cout << "\n \n Gain\n ***************************\n \n";
        std::cout << a11 << " " << a12 << " " << a13 << std::endl;
        std::cout << a21 << " " << a22 << " " << a23 << std::endl;
        std::cout << a31 << " " << a32 << " " << a33 << std::endl;
        std::cout << "\n \n \n ***************************\n \n";


        while (true) {
            std::cout << "*****************************************************************" << endl;
            std::cout << "Control Period = " << sx << endl;

            auto start = std::chrono::high_resolution_clock::now();
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> durationa = end - start;

            float prevValue1 = 0;
            float prevValue2 = 0;
            float prevValue3 = 0;


            while (durationa.count() < 4) {
                auto end = std::chrono::high_resolution_clock::now();
                durationa = end - start;

                // Check if the new values are different from the previous ones
                if (sharedData->values[sm1] != prevValue1) {
                    int1 << sharedData->values[sm1] << std::endl;

                    avg.taskexec1.push_back(1 * sharedData->values[sm1] / (sharedData->newperiods[proc11] * 1000));
                    avg.taskexec11.push_back(sharedData->values[sm1]);
                    prevValue1 = sharedData->values[sm1];
                }
                if (sharedData->values[sm2] != prevValue2) {
                    int2 << sharedData->values[sm2] << std::endl;
                    avg.taskexec2.push_back(1 * sharedData->values[sm2] / (sharedData->newperiods[proc22] * 1000));

                    avg.taskexec22.push_back(sharedData->values[sm2]);
                    prevValue2 = sharedData->values[sm2];
                }

                if (sharedData->values[sm3] != prevValue3) {
                    int3 << sharedData->values[sm3] << std::endl;
                    avg.taskexec3.push_back(1 * sharedData->values[sm3] / (sharedData->newperiods[proc33] * 1000));

                    avg.taskexec33.push_back(sharedData->values[sm3]);
                    prevValue3 = sharedData->values[sm3];
                }
            }
            float rtr1 = calculatePercentage(avg.taskexec11, sharedData->newperiods[proc11] * 1000);
            float rtr2 = calculatePercentage(avg.taskexec22, sharedData->newperiods[proc22] * 1000);
            float rtr3 = calculatePercentage(avg.taskexec33, sharedData->newperiods[proc33] * 1000);
            rtrdeadline1 << rtr1 << endl;
            rtrdeadline2 << rtr2 << endl;
            rtrdeadline3 << rtr3 << endl;

            if (neworkload == true) {
                goto start;
                std::cout << "Go to start \t" << endl;
            }


            int1 << "Control Period \n";
            int2 << "Control Period \n";
            int3 << "Control Period \n";
            setpointt1 << sharedData->newperiods[proc11] * 1000 << endl;
            setpointt2 << sharedData->newperiods[proc22] * 1000 << endl;
            setpointt3 << sharedData->newperiods[proc33] * 1000 << endl;

            size_t windowSize = 8;  // Choose an appropriate window size
            applyMedianFilter(avg.taskexec1, windowSize);
            applyMedianFilter(avg.taskexec2, windowSize);
            applyMedianFilter(avg.taskexec3, windowSize);
            float avg_task1 = avg.calculateAverage(avg.taskexec1);
            float avg_task2 = avg.calculateAverage(avg.taskexec2);
            float avg_task3 = avg.calculateAverage(avg.taskexec3);
            float avg_task4 = avg.calculateAverage(avg.taskexec11);
            float avg_task5 = avg.calculateAverage(avg.taskexec22);
            float avg_task6 = avg.calculateAverage(avg.taskexec33);

            s1 = avg_task1 / (1);
            s2 = avg_task2 / (1);
            s3 = avg_task3 / (1);
            std::cout << "RT 1 = \t" << avg_task4 << "\t RT2 = \t" << avg_task5
                << "\t RT3 = \t" << avg_task6 << endl;
            std::cout << "Period 1 = \t" << sharedData->newperiods[proc11] * 1 << "\t Period 2 = \t" << sharedData->newperiods[proc22] * 1 << "\t Period 3 = \t" << sharedData->newperiods[proc33] * 1 << endl;
            std::cout << "S1 = \t" << s1 << "\tS2 = \t" << s2 << "\t S3 = " << s3 << endl;


            avg.taskexec1.clear();
            avg.taskexec2.clear();
            avg.taskexec3.clear();
            avg.taskexec11.clear();
            avg.taskexec22.clear();
            avg.taskexec33.clear();

            slack1 << s1 << endl;
            slack2 << s2 << endl;
            slack3 << s3 << endl;

            sep = setpoint1 - s1;
            sep2 = setpoint2 - s2;
            sep3 = setpoint3 - s3;
            std::cout << "Error1 = " << sep * 1 << "\tError2 = " << sep2 * 1 <<
                "\tError3 = " << sep3 * 1 << endl;


            s1 = (sep * a11) + (sep2 * a12) + (sep3 * a13);
            s2 = (sep * a21) + (sep2 * a22) + (sep3 * a23);
            s3 = (sep * a31) + (sep2 * a32) + (sep3 * a33);
            error1 = sep;
            error2 = sep2;
            error3 = sep3;
            durationarrra << sx << endl;
            new1 = sharedData->newperiods[proc11];
            new2 = sharedData->newperiods[proc22];
            new3 = sharedData->newperiods[proc33];
            float period1 = new1;
            float period2 = new2;
            float period3 = new3;
            std::cout << " Delta1 = " << s1 * 1 << "\tDelta2 = " << s2 * 1
                << "\tDelta3 = " << s3 * 1 << endl;
            std::cout << "Old Period 1 = " << new1 << "\tOld Period 2 = " << new2
                << "\tOld Period 3 = " << new3 << endl;

            //new1=period1+s1/(period2+period3);
            //new2=period2+s2/(period1+period3);
            //new3=period3+s3/(period2+period1);

            new1 = (period1) / (1 + (period1 * (period2 + period3) * s1));
            new2 = (period2) / (1 + (period2 * (period1 + period3) * s2));
            new3 = (period3) / (1 + (period3 * (period1 + period2) * s3));


            std::cout << "New Period = " << new1 << "\tPeriod2 = " << new2 <<
                "\t New period 3 = " << new3 << endl;
            std::cout << "*****************************************************************" << endl;
            std::cout << endl;

            //sharedData->newperiods[proc11]=new1;
            //sharedData->newperiods[proc22]=new2;
            sharedData->newperiods[proc11] = new1;
            sharedData->newperiods[proc22] = new2;
            sharedData->newperiods[proc33] = new3;

            cpt1 << sharedData->newperiods[proc11] * 1000 << endl;
            cpt2 << sharedData->newperiods[proc22] * 1000 << endl;
            cpt3 << sharedData->newperiods[proc33] * 1000 << endl;

            signalReceived = false;
            signalReceived2 = false;
            signalReceived3 = false;
            //kill(pidb, SIGHUP);

            count = 0;
            sx = sx + 1;


        }


    } else if (tasks == 4) {
        signal(mappingTable[task1_str], signalHandler3);
        signal(mappingTable[task2_str], signalHandler3);
        signal(mappingTable[task3_str], signalHandler3);
        signal(mappingTable[task4_str], signalHandler3);

        std::ifstream fpt("logs/setpoints.txt");

        if (!fpt.is_open()) {
            std::cerr << "Error: Could not open the file." << std::endl;
            return 1; // Return an error code
        }

        fpt >> setpoint1 >> setpoint2 >> setpoint3 >> setpoint4;

        // Close the file stream
        fpt.close();

        // Output the values read from the file
        std::cout << "sepa: " << setpoint1 << std::endl;
        std::cout << "sepb: " << setpoint2 << std::endl;
        std::cout << "sepc: " << setpoint3 << std::endl;
        std::cout << "sepd: " << setpoint4 << std::endl;


        double totautil;
        float sep;
        float sep2;
        float sep3;
        float sep4;

        float a11;
        float a12;
        float a13;
        float a14;
        float a21;
        float a22;
        float a23;
        float a24;
        float a31;
        float a32;
        float a33;
        float a34;
        float a41;
        float a42;
        float a43;
        float a44;

        float s1;
        float s2;
        float s3;
        float s4;
        float new1, new2, new3, new4;
        new1 = 0;
        new2 = 0;
        new3 = 0;
        new4 = 0;
        float sx = 0;

        float workloadchange = 0;

      
        
        a11 = 2.6968 * 10;
        a12 = 1.3968 / 10;
        a13 = 0.3968 / 10;
        a14 = 0.3968 / 10;
        a21 = 0.3968 / 10;
        a22 = 2.6968 * 10;
        a23 = 0.3968 / 10;
        a24 = 0.3968 / 10;
        a31 = 0.2976 / 10;
        a32 = 0.2976 / 10;
        a33 = 2.6976 * 10;
        a34 = 0.2976 / 10;
        a41 = 0.2976 / 10;
        a42 = 0.2976 / 10;
        a43 = 0.2976 / 10;
        a44 = 2.6976 * 10;


        std::cout << "****************************************************\n";
        std::cout << "****************************************************\n";
        std::cout << "****************************************************\n";
        std::cout << "****************************************************\n";
        std::cout << "****************************************************\n";

        average avg;

        //kill(pida, SIGHUP);
        //kill(pidb, SIGHUP);
        //kill(pidc, SIGHUP);
        //kill(pidc, SIGHUP);
        std::cout << "\n\n Gain\n ***************************\n\n";
        std::cout << a11 << " " << a12 << " " << a13 << " " << a14 << std::endl;
        std::cout << a21 << " " << a22 << " " << a23 << " " << a24 << std::endl;
        std::cout << a31 << " " << a32 << " " << a33 << " " << a34 << std::endl;
        std::cout << a41 << " " << a42 << " " << a43 << " " << a44 << std::endl;
        std::cout << "\n\n\n ***************************\n\n";
        usleep(2000000); // 2,000,000 microseconds = 2 seconds


        while (true) {
            std::cout << "*****************************************************************" << endl;
            std::cout << "Control Period = " << sx << endl;

            auto start = std::chrono::high_resolution_clock::now();
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> durationa = end - start;

            float prevValue1 = 0;
            float prevValue2 = 0;
            float prevValue3 = 0;
            float prevValue4 = 0;
            while (durationa.count() < 5) {
                auto end = std::chrono::high_resolution_clock::now();
                durationa = end - start;

                // Check if the new values are different from the previous ones
                if (sharedData->values[sm1] != prevValue1) {
                    int1 << sharedData->values[sm1] << std::endl;
                    avg.taskexec1.push_back(1 * sharedData->values[sm1] / (sharedData->newperiods[proc11] * 1000));
                    avg.taskexec11.push_back(sharedData->values[sm1]);
                    prevValue1 = sharedData->values[sm1];
                }
                if (sharedData->values[sm2] != prevValue2) {
                    int2 << sharedData->values[sm2] << std::endl;
                    avg.taskexec2.push_back(1 * sharedData->values[sm2] / (sharedData->newperiods[proc22] * 1000));

                    avg.taskexec22.push_back(sharedData->values[sm2]);
                    prevValue2 = sharedData->values[sm2];
                }

                if (sharedData->values[sm3] != prevValue3) {
                    int3 << sharedData->values[sm3] << std::endl;
                    avg.taskexec3.push_back(1 * sharedData->values[sm3] / (sharedData->newperiods[proc33] * 1000));

                    avg.taskexec33.push_back(sharedData->values[sm3]);
                    prevValue3 = sharedData->values[sm3];
                }

                if (sharedData->values[sm4] != prevValue4) {
                    int4 << sharedData->values[sm4] << std::endl;
                    avg.taskexec4.push_back(1 * sharedData->values[sm4] / (sharedData->newperiods[proc44] * 1000));

                    avg.taskexec44.push_back(sharedData->values[sm4]);
                    prevValue4 = sharedData->values[sm4];
                }


            }

            float rtr1 = calculatePercentage(avg.taskexec11, sharedData->newperiods[proc11] * 1000);
            float rtr2 = calculatePercentage(avg.taskexec22, sharedData->newperiods[proc22] * 1000);
            float rtr3 = calculatePercentage(avg.taskexec33, sharedData->newperiods[proc33] * 1000);
            float rtr4 = calculatePercentage(avg.taskexec44, sharedData->newperiods[proc44] * 1000);
            rtrdeadline1 << rtr1 << endl;
            rtrdeadline2 << rtr2 << endl;
            rtrdeadline3 << rtr3 << endl;
            rtrdeadline4 << rtr4 << endl;


            int1 << "Control Period \n";
            int2 << "Control Period \n";
            int3 << "Control Period \n";
            int4 << "Control Period \n";
            setpointt1 << sharedData->newperiods[proc11] * 1000 << endl;
            setpointt2 << sharedData->newperiods[proc22] * 1000 << endl;
            setpointt3 << sharedData->newperiods[proc33] * 1000 << endl;
            setpointt4 << sharedData->newperiods[proc44] * 1000 << endl;
            size_t windowSize = 8;  // Choose an appropriate window size
            applyMedianFilter(avg.taskexec1, windowSize);
            applyMedianFilter(avg.taskexec2, windowSize);
            applyMedianFilter(avg.taskexec3, windowSize);
            applyMedianFilter(avg.taskexec4, windowSize);
            float avg_task1 = avg.calculateAverage(avg.taskexec1);
            float avg_task2 = avg.calculateAverage(avg.taskexec2);
            float avg_task3 = avg.calculateAverage(avg.taskexec3);
            float avg_task4 = avg.calculateAverage(avg.taskexec4);
            float avg_task5 = avg.calculateAverage(avg.taskexec11);
            float avg_task6 = avg.calculateAverage(avg.taskexec22);
            float avg_task7 = avg.calculateAverage(avg.taskexec33);
            float avg_task8 = avg.calculateAverage(avg.taskexec44);

            s1 = avg_task1 / (1);
            s2 = avg_task2 / (1);
            s3 = avg_task3 / (1);
            s4 = avg_task4 / (1);
            std::cout << "RT 1 = \t" << avg_task5 << "\t RT2 = \t" << avg_task6
                << "\t RT3 = \t" << avg_task7 << "\t RT4 = \t" << avg_task8 << endl;
            std::cout << "Period 1 = \t" << sharedData->newperiods[proc11] * 1 << "\t Period 2 = \t" << sharedData->newperiods[proc22] * 1 << "\t Period 3 = \t" << sharedData->newperiods[proc33] * 1 << "\t Period 4 = \t" << sharedData->newperiods[proc44] * 1 << endl;
            std::cout << "S1 = \t" << s1 << "\tS2 = \t" << s2 << "\t S3 = " << s3 <<
                "\t S4 = " << s4 << endl;


            avg.taskexec1.clear();
            avg.taskexec2.clear();
            avg.taskexec3.clear();
            avg.taskexec4.clear();
            avg.taskexec11.clear();
            avg.taskexec22.clear();
            avg.taskexec33.clear();
            avg.taskexec44.clear();

            slack1 << s1 << endl;
            slack2 << s2 << endl;
            slack3 << s3 << endl;
            slack4 << s4 << endl;

            sep = setpoint1 - s1;
            sep2 = setpoint2 - s2;
            sep3 = setpoint3 - s3;
            sep4 = setpoint4 - s4;
            std::cout << "Error1 = " << sep * 1
                << "\tError2 = " << sep2 * 1
                << "\tError3 = " << sep3 * 1
                << "\tError4 = " << sep4 * 1
                << std::endl;


            s1 = (sep * a11) + (sep2 * a12) + (sep3 * a13) + (sep4 * a14);
            s2 = (sep * a21) + (sep2 * a22) + (sep3 * a23) + (sep4 * a24);
            s3 = (sep * a31) + (sep2 * a32) + (sep3 * a33) + (sep4 * a34);
            s4 = (sep * a41) + (sep2 * a42) + (sep3 * a43) + (sep4 * a44);

            durationarrra << sx << endl;
            new1 = sharedData->newperiods[proc11];
            new2 = sharedData->newperiods[proc22];
            new3 = sharedData->newperiods[proc33];
            new4 = sharedData->newperiods[proc44];
            float period1 = new1;
            float period2 = new2;
            float period3 = new3;
            float period4 = new4;
            std::cout << "Delta1 = " << s1 * 1
                << "\tDelta2 = " << s2 * 1
                << "\tDelta3 = " << s3 * 1
                << "\tDelta4 = " << s4 * 1
                << std::endl;
            std::cout << "Old Period 1 = " << new1
                << "\tOld Period 2 = " << new2
                << "\tOld Period 3 = " << new3
                << "\tOld Period 4 = " << new4
                << std::endl;

            //new1=period1+s1/(period2+period3);
            //new2=period2+s2/(period1+period3);
            //new3=period3+s3/(period2+period1);

            new1 = (period1) / (1 + (period1 * (period2 + period3 + period4) * s1));
            new2 = (period2) / (1 + (period2 * (period1 + period3 + period4) * s2));
            new3 = (period3) / (1 + (period3 * (period2 + period1 + period4) * s3));
            new4 = (period4) / (1 + (period4 * (period2 + period3 + period1) * s4));


            std::cout << "New Period 1 = " << new1
                << "\tPeriod 2 = " << new2
                << "\tNew Period 3 = " << new3
                << "\tPeriod 4 = " << new4
                << std::endl;
            std::cout << endl;

            //sharedData->newperiods[proc11]=new1;
            //sharedData->newperiods[proc22]=new2;
            sharedData->newperiods[proc11] = new1;
            sharedData->newperiods[proc22] = new2;
            sharedData->newperiods[proc33] = new3;
            sharedData->newperiods[proc44] = new4;

            cpt1 << sharedData->newperiods[proc11] * 1000 << endl;
            cpt2 << sharedData->newperiods[proc22] * 1000 << endl;
            cpt3 << sharedData->newperiods[proc33] * 1000 << endl;
            cpt4 << sharedData->newperiods[proc44] * 1000 << endl;

            signalReceived = false;
            signalReceived2 = false;
            signalReceived3 = false;
            signalReceived4 = false;
            //kill(pidb, SIGHUP);

            count = 0;
            sx = sx + 1;


        }


    }



    int1.close();
    int2.close();
    int3.close();
    int4.close();
    cpt1.close();
    cpt2.close();
    cpt3.close();
    cpt4.close();
    slack1.close();
    slack2.close();
    slack3.close();
    slack4.close();
    durationarrra.close();
    setpointt1.close();
    setpointt2.close();
    setpointt3.close();
    setpointt4.close();
}
