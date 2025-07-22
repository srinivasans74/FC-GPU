#include <iostream>
#include <cstdlib>
#include <vector>
#include <ctime>
#include <cmath>
#include <sstream> // Add this line for stringstream
#include <chrono>
#include <iostream>
#include<vector>
#include <csignal>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <cstdlib> // Include the <cstdlib> header for the exit() function
#include <semaphore.h>
#include <unistd.h>
#include<csignal>
#include <string.h>
#include <float.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <fstream>
#include <thread>
#include <cassert>
#include <iomanip>
using namespace std::chrono;
#include <cassert>
using namespace std;
#include <future>
#include <queue>
#include <unordered_map>
#include <iomanip>
#include <iostream>
#include <vector>
#include <deque>
int width = 8; // Width of each field
int precision = 2; // Decimal precision




struct average
{
    
    vector<float>taskexec1;
    vector<float>taskexec2;
    vector<float>taskexec3;
    vector<float>taskexec4;

    vector<float>taskexec11;
    vector<float>taskexec22;
    vector<float>taskexec33;
    vector<float>taskexec44;

 float calculateAverage(const std::vector<float>& values) {
        if (values.empty()) {
            return 0.0; // return 0 if the vector is empty to avoid division by zero
        }
        float sum = 0.0;
        for (float value : values) {
            sum += value;
        }
        return sum / values.size();
    }


};


#define SIGUSR3 30
#define SIGUSR4 31
int sm1 = 0, sm11 = 0, sm2 = 0, sm22 = 0, sm3 = 0, sm33 = 0, sm4 = 0, sm44 = 0;

struct SharedData {
    float values[8]; 
    float newperiods[4]; 
};

SharedData* sharedData;
int proc11;
int proc22;
int proc33;
int proc44;



