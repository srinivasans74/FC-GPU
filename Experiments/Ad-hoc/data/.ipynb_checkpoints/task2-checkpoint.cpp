#include "shared_data.h"


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
    std::cout<<"PID PARENT T2= "<<pids<<endl;

}





int main(int argc, char** argv) {
 //   std::cout << "T2" << std::endl;
    Timer clock;  // Timer ticks upon construction.
    Timer clock1;  // Timer ticks upon construction.
    Timer clock2;  // Timer ticks upon construction.
    Timer clock3;  // Timer ticks upon construction.    
    signal(SIGHUP, signalHandler);
    
    key_t key = ftok("shmfile", 65);
    int shmid = shmget(key, sizeof(SharedData), 0666);
    SharedData* sharedData = (SharedData*)shmat(shmid, nullptr, 0);
    
    key_t keya = ftok("shmfile", 75);
    int shmida = shmget(keya, sizeof(proc), 0666);
    proc* proca = (proc*)shmat(shmida, nullptr, 0);

    
    std::string command= "../../benchmarks/stereoDisparity";
  //  std::string command = "../../benchmarks/lud_cuda  -s 12000";
    //command= "../../benchmarks/mttkrp";
    std::string nullDevice = " > /dev/null 2>&1"; // On Unix-like systems
    std::string fullCommand = command + nullDevice;
    // std::string fullCommand = command;
    auto start = high_resolution_clock::now();
    int x=4;
  //  std::cout << "T22" << std::endl;
      for (int t=0;t<x;t++) {
        std::system(fullCommand.c_str());
    }
    std::cout<<"Break2"<<endl;
    auto stop = high_resolution_clock::now();
    auto duration2 = duration_cast<microseconds>(stop - start);
    double period = std::chrono::duration<double>(duration2).count();
    period=period/x;
    sharedData->values[3]=period;
    sharedData->values[1]=std::stod(argv[1]);
    std::cout << "p2" <<period<< std::endl;
    //std::cout << "T23" <<duration2<< std::endl;
    //kill(pids, SIGHUP);
    init();
    kill(pids, SIGUSR2);
    while (!signalReceived) {
                usleep(1000);
            }
    signalReceived = false;

    //float period = ((duration2.count()/2) * 1.0e-6) -1;
    //double period=std::stod(argv[2]);
    //std::cout<<"Period ="<<period<<endl;
    double step1 = 0.50;
    double step2 = 0.63;
    std::string firstArgument = argv[1];
    double setpoint = std::stod(firstArgument); 
    period=std::stod(argv[2]);
    std::string firstArgument1 = argv[3];
    float sep=0;
    float s1=0;
    int iteration = std::stod(firstArgument1);  // Convert string to double
    //system("clear");

    //std::cout<<"Tune 1 ="<<tune1<<"Tune 2 ="<<tune2<<endl<<"\n";
   std::ofstream outputFile("logs/logt2.txt");
   // outputFile << std::fixed << std::setprecision(6);
   // outputFile << "Tune 1 ="<<tune1<<"Tune 2 ="<<tune2<<endl<<"\n";
   clock1.tick();
    clock3.tick();
    int i=0;
    vector<float> slackarray; 
    vector<float> periodarray; 
     vector<float> durationa; 
     vector<float> timearray; 

    std::future<void> asyncTask ;
   while(true) 
       {
        clock1.tock();
        double releasetime = ((clock1.duration().count() / 1000));
         outputFile<<"Task released at  time = "<<releasetime<<endl;
         double duration = (floor(clock1.duration().count() / 1000));
        if(duration>iteration) 
        {
               // std::cout<<"Finished execution t2 at jhxn at "<<clock3.duration().count() / 1000<<endl;
                std::cout<<signalReceived<<endl;
                break;   
        }
           if (asyncTask.valid()) {
        asyncTask.get();
      }

       double init = period;
       periodarray.push_back(period);
      asyncTask = std::async(std::launch::async, [&]() {
            clock.tick();
            std::system(fullCommand.c_str());
            clock.tock();
            clock3.tock();
            float check = (floor(clock.duration().count() / 1000)) / init;
            timearray.push_back((clock3.duration().count() / 1000));
            slackarray.push_back(check);
            duration = (floor(clock.duration().count() / 1000));
            durationa.push_back(duration);
            sharedData->values[1] = check;
            sharedData->values[3] = duration;
            //kill(pids, SIGHUP);
            kill(pids, SIGUSR2);

            while (!signalReceived) {
                usleep(1000);
            }
            signalReceived = false;
            period = period + sharedData->values[1];
        });
       
        std::chrono::milliseconds sleep_duration(static_cast<long long>(std::max(0.0, (init) * 1000.0)));
        std::this_thread::sleep_for(sleep_duration);
        //asyncTask.get();


       i=i+1;
       
       
       
    }
   std::cout<<signalReceived<<endl;
    proca->values[1]=1;
    signalReceived=true;

    if (asyncTask.valid()) {
    asyncTask.get();
}
   //   proca->values[1]=1;
    signalReceived=true;
       clock3.tock();
    std::cout<<"Finished execution t2 at "<<clock3.duration().count() / 1000<<endl;
     while (!signalReceived) {
                usleep(1000);
             if (proca->values[1]==0)
             {
             signalReceived = false;
             }}
     

   
       FILE *file = fopen("logs/s2.txt", "w");
    for (auto a = slackarray.begin(); a != slackarray.end(); ++a) {
        fprintf(file, "%f\n", *a);
    }

    fclose(file);

    FILE *file1 = fopen("logs/pt2.txt", "w");
    for (auto a = periodarray.begin(); a != periodarray.end(); ++a) {
        fprintf(file1, "%f\n", *a);
    }

    fclose(file1);
    
    
      FILE *file2 = fopen("logs/et2.txt", "w");
      for(auto a=durationa.begin(); a!=durationa.end();a++){
        fprintf(file2, "%f\n", *a);
    }
    fclose(file2); 
    
            
   FILE *file3 = fopen("logs/rt2.txt", "w");
      for(auto a=timearray.begin(); a!=timearray.end();a++){
        fprintf(file3, "%f\n", *a);
    }
    fclose(file3);
    
    FILE *file4 = fopen("logs/t2.txt", "w");
      for(auto a=timearray.begin(); a!=timearray.end();a++){
        fprintf(file3, "%f\n", setpoint);
    }
    fclose(file4);
    
    
    return 0;
}
