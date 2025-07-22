#include "shared_data.h"

class Queue {
private:
    std::queue<int> elements;

public:
    void push(int value) {
        elements.push(value);
    }

    int pop() {
        if (elements.empty()) {
            std::cerr << "Queue is empty. Cannot pop.\n";
            return -1; // Return some error value
        }
        int value = elements.front();
        elements.pop();
        return value;
    }

    int front() const {
        if (elements.empty()) {
            std::cerr << "Queue is empty. No front element.\n";
            return -1; // Return some error value
        }
        return elements.front();
    }

    bool isEmpty() const {
        return elements.empty();
    }
};
          
                   
template <class DT = std::chrono::milliseconds,
          class ClockT = std::chrono::high_resolution_clock>
              
class Timer
{
    using timep_t = decltype(ClockT::now());
    
    timep_t _start = ClockT::now();
    timep_t _end = {};

public:
    void tick() { 
        _end = timep_t{};
        _start = ClockT::now(); 
    }
    
    void tock() {
        _end = ClockT::now(); 
    }
    
    template <class duration_t = DT>
    auto duration() const { 
        // Use gsl_Expects if your project supports it.
        assert(_end != timep_t{} && "Timer must toc before reading the time"); 
        return std::chrono::duration_cast<duration_t>(_end - _start); 
    }
};

              
bool signalReceived = false;
int pids=0;
// Signal handler function
void signalHandler(int signum) {
    if (signum == SIGHUP) {
        signalReceived = true;
    }
}              
         

void init()
{
    FILE *file = fopen("logs/mainpid.txt", "r");
    int value;
    fscanf(file, "%d", &value);
    pids=value;
    fclose(file);
    std::cout<<"PID PARENT T1= "<<pids<<endl;

}


int main(int argc, char** argv) {
    std::cout << "T1" << std::endl;
    Timer clock;  // Timer ticks upon construction.
    Timer clock1;  // Timer ticks upon construction.
    Timer clock2;  // Timer ticks upon construction.
    Timer clock3;  // Timer ticks upon construction.
    Queue q1;
    signal(SIGHUP, signalHandler);
    std::cout<<"T11";
    key_t key = ftok("shmfile", 65);
    int shmid = shmget(key, 1024, 0666 | IPC_CREAT);
    SharedData* sharedData = (SharedData*)shmat(shmid, nullptr, 0);
    std::cout << "T121" << std::endl;
    key_t keya = ftok("shmfile", 75);
    int shmida = shmget(keya, 1024, 0666 | IPC_CREAT);
    proc* proca = (proc*)shmat(shmida, nullptr, 0);
    
    std::string command = "../../benchmarks/lud_cuda  -s 12000";
    std::string nullDevice = " > /dev/null 2>&1"; // On Unix-like systems
    std::string fullCommand = command + nullDevice;
    
    
    auto start = high_resolution_clock::now();
    //std::cout << "Warmup t1" << std::endl;
    int x=4;
    for (int t=0;t<x;t++) {
        std::system(fullCommand.c_str());
    }
    std::cout<<"Warmup t1 Completed"<<endl;
    auto stop = high_resolution_clock::now();
    auto duration2 = duration_cast<microseconds>(stop - start);
    double period = std::chrono::duration<double>(duration2).count();
    period=period/x;
    
    std::cout << "p1" <<period<< std::endl;
    sharedData->values[2]=period;
    sharedData->values[0]=std::stod(argv[1]);
    init();
    kill(pids, SIGUSR1);
    while (!signalReceived) {
                usleep(1000);
            }
    signalReceived = false;
    
    
    
    period=std::stod(argv[2]);
    std::cout<<"Period T1 =" <<period<<endl;
    double range = std::stod(argv[1]); 
    int iteration = std::stod(argv[3]); 
 
    std::ofstream outputFile("logs/logt1.txt");
    vector<float> slackarray; 
    vector<float> periodarray; 
    vector<float> durationa; 
    vector<float> timearray; 

    clock1.tick();
    clock2.tick();
    clock3.tick();
    
    int i=0;
    std::future<void> asyncTask ;
   while(true) 
       {
       clock1.tock();
       clock2.tock();
       double releasetime = (floor(clock2.duration().count() / 1000));
       outputFile<<"Task released at  time = "<<releasetime<<endl;
       double duration = (floor(clock1.duration().count() / 1000));
        if(duration>=iteration) 
        {
            
            break;   
        }
       if(i!=1)
       {
         clock.tick();  
       }
       
      // std::cout<<"Duration k "<<duration<<endl;
           if (asyncTask.valid()) {
        asyncTask.get();
      }
    
       double init = period;
       periodarray.push_back(period);
      asyncTask = std::async(std::launch::async, [&]() {
        if(i==0)
        {
        // i=i+1;   
        clock.tick();
        }
        std::system(fullCommand.c_str());
        clock.tock();
        clock3.tock();
        float check = (floor(clock.duration().count() / 1000)) / init;
        slackarray.push_back(check);
        timearray.push_back((clock3.duration().count() / 1000));
        duration = (floor(clock.duration().count() / 1000));
        durationa.push_back(duration);
        sharedData->values[0] = check;
        sharedData->values[2] = duration;
        //kill(pids, SIGHUP);
        kill(pids, SIGUSR1);

            while (!signalReceived) {
                usleep(1000);
            }
            signalReceived = false;
            period = period + sharedData->values[0];
        });
        std::chrono::milliseconds sleep_duration(static_cast<long long>(std::max(0.0, (init) * 1000.0)));
        std::this_thread::sleep_for(sleep_duration);
       // asyncTask.get();


       i=i+1;
       
       
    }
    if (asyncTask.valid()) {
    asyncTask.get();
}
    
    
    signalReceived=true;
    proca->values[0]=1;
    clock3.tock();
    std::cout<<"Finished execution t1 at "<<clock3.duration().count() / 1000<<endl;
    
    
     while (!signalReceived) {
                usleep(1000);
             if (proca->values[0]==0)
             {
             signalReceived = false;
             }
     }
    shmdt(sharedData);
    shmdt(proca);
    outputFile.close();
    
    
    
    
    
    
    
    
    
    
    
    
       FILE *file = fopen("logs/s1.txt", "w");
    for (auto a = slackarray.begin(); a != slackarray.end(); ++a) {
        fprintf(file, "%f\n", *a);
    }

    fclose(file);

    FILE *file1 = fopen("logs/pt1.txt", "w");
    for (auto a = periodarray.begin(); a != periodarray.end(); ++a) {
        fprintf(file1, "%f\n", *a);
    }

    fclose(file1);
    
    
      FILE *file2 = fopen("logs/et1.txt", "w");
      for(auto a=durationa.begin(); a!=durationa.end();a++){
        fprintf(file2, "%f\n", *a);
    }
    fclose(file2); 
    
            
   FILE *file3 = fopen("logs/rt1.txt", "w");
      for(auto a=timearray.begin(); a!=timearray.end();a++){
        fprintf(file3, "%f\n", *a);
    }
    fclose(file3);
    
    FILE *file4 = fopen("logs/t1.txt", "w");
      for(auto a=timearray.begin(); a!=timearray.end();a++){
        fprintf(file3, "%f\n", range);
    }
    fclose(file4);
    
    
    
    
    shmctl(shmid,  IPC_RMID, NULL);
    shmctl(shmida,  IPC_RMID, NULL);
    kill(pids, SIGTERM);
    return 0;
}
