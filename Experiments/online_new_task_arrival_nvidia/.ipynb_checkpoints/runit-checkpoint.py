import os
import subprocess
import time
from time import sleep
import sys
import signal
if __name__=='__main__':
    process_list=[]  
    print(sys.argv[1])
    #print(sys.argv[2])
    #print(sys.argv[3])

    if int(sys.argv[1]) == 4:
        t1=sys.argv[2]
        t2=sys.argv[3]
        t3=sys.argv[4]
        t4=sys.argv[5]
        start=time.time()
        print(t1)
        print(t2)
        print(t3)
        with open('logs/task_values.txt', 'w') as file:
            file.write(f"{t1}\n")
            file.write(f"{t2}\n")
            file.write(f"{t3}\n")
        file.close
        command1=["./"+str(t1),sys.argv[6],sys.argv[7] ,sys.argv[14]]
        command2=["./"+str(t2),sys.argv[8],sys.argv[9],sys.argv[14]]
        command3=["./"+str(t3),sys.argv[10],sys.argv[11],sys.argv[14]]
        process_list.append(subprocess.Popen(command1))
        process_list.append(subprocess.Popen(command2))
        process_list.append(subprocess.Popen(command3))
        pid = process_list[0].pid
        pid2 = process_list[1].pid
        pid3 = process_list[2].pid
        command4=["./server",str(len(process_list)),str(pid),str(pid2),str(pid3),sys.argv[6],sys.argv[8],sys.argv[10]]
        process_list.append(subprocess.Popen(command4))
        pid4 = process_list[3].pid

     #   process_list.append(subprocess.Popen(command3))
    #    pid = process_list[0].pid
    #    pid2 = process_list[1].pid
       # pid3 = process_list[2].pid
        print(command1)
        print(command2)
        print(command3)
        print(command4)
        timeout = float(sys.argv[14])
        switch=0
        while process_list:
            for process in process_list[:]: 
                if process.poll() is not None:
                    process_list.remove(process)  # Remove completed process
                else:                    
                    elapsed_time = time.time() - start
                    if(elapsed_time>=0.5*timeout and switch==0):
                        print('\n *************** \n')
                        print('Workload change \n \n \n')
                        print('*************** \n')
                        with open('logs/task_values.txt', 'w') as file:
                            file.write(f"{t1}\n")
                            file.write(f"{t2}\n")
                            file.write(f"{t3}\n")
                            file.write(f"{t4}\n")
                        file.close
                        command5=["./"+str(t4),sys.argv[10],sys.argv[11],sys.argv[14]]
                        print(command5)
                        process_list.append(subprocess.Popen(command5))  
                        with open('logs/setpoints.txt', 'w') as file:
                            file.write(f"{sys.argv[6]}\n")
                            file.write(f"{sys.argv[8]}\n")
                            file.write(f"{sys.argv[10]}\n")
                            file.write(f"{sys.argv[12]}\n")

                        os.kill(pid4, signal.SIGINT)  
                        switch=switch+1
                        
                    if elapsed_time >= timeout:
                        process.kill()  # Kill the process if it exceeds the time limit
                        process_list.remove(process)  # Remove the killed process
                    else:
                        time.sleep(0.1)  
        
        
       #

        
        #command4=["./server",str(len(process_list)),str(pid),str(pid2),str(pid3),sys.argv[5],sys.argv[7],sys.argv[9]]
        #print(command4)
        #process_list.append(subprocess.Popen(command4))
       
  