import os
import subprocess
import time
import sys
from time import sleep

if __name__ == '__main__':
    process_list = []
    print(sys.argv[1])

    if int(sys.argv[1]) == 2:
        t1 = sys.argv[2]
        t2 = sys.argv[3]
        start = time.time()
        print(t1)
        print(t2)
        with open('logs/task_values.txt', 'w') as file:
            file.write(f"{t1}\n")
            file.write(f"{t2}\n")
        command1 = ["./" + str(t1), sys.argv[4], sys.argv[5], sys.argv[8]]
        command2 = ["./" + str(t2), sys.argv[6], sys.argv[7], sys.argv[8]]
        print(command1)
        print(command2)
        process_list.append(subprocess.Popen(command1))
        time.sleep(1)
        process_list.append(subprocess.Popen(command2))
        time.sleep(2)
        pid = process_list[0].pid
        pid2 = process_list[1].pid
        command3 = ["./server", str(len(process_list)), str(pid), str(pid2), sys.argv[4], sys.argv[6]]
        process_list.append(subprocess.Popen(command3))
        print(command3)
        timeout = float(sys.argv[8]) + 5
        while process_list:
            for process in process_list[:]:
                if process.poll() is not None:
                    process_list.remove(process)
                else:
                    elapsed_time = time.time() - start
                    if elapsed_time >= timeout:
                        process.kill()
                        process_list.remove(process)
                    else:
                        time.sleep(0.1)
    elif int(sys.argv[1])==3:
        t1 = sys.argv[2]
        t2 = sys.argv[3]
        t3 = sys.argv[4]
        start = time.time()
        print(t1)
        print(t2)
        print(t3)
        with open('logs/task_values.txt', 'w') as file:
            file.write(f"{t1}\n")
            file.write(f"{t2}\n")
            file.write(f"{t3}\n")
        command1 = ["./" + str(t1), sys.argv[5], sys.argv[6], sys.argv[11]]
        command2 = ["./" + str(t2), sys.argv[7], sys.argv[8], sys.argv[11]]
        command3 = ["./" + str(t3), sys.argv[9], sys.argv[10], sys.argv[11]]
        print(command1)
        print(command2)
        print(command3)
        process_list.append(subprocess.Popen(command1))
        process_list.append(subprocess.Popen(command2))
        process_list.append(subprocess.Popen(command3))
        pid = process_list[0].pid
        pid2 = process_list[1].pid
        pid3 = process_list[2].pid
        command4 = ["./server", str(len(process_list)), str(pid), str(pid2), str(pid3), sys.argv[5], sys.argv[7],
                    sys.argv[9]]
        sleep(2)
        process_list.append(subprocess.Popen(command4))
        print(command4)
        timeout = float(sys.argv[11]) + 5
        while process_list:
            for process in process_list[:]:
                if process.poll() is not None:
                    process_list.remove(process)
                else:
                    elapsed_time = time.time() - start
                    if elapsed_time >= timeout:
                        process.kill()
                        process_list.remove(process)
                    else:
                        time.sleep(0.1)
    elif int(sys.argv[1])==4:
        t1 = sys.argv[2]
        t2 = sys.argv[3]
        t3 = sys.argv[4]
        t4 = sys.argv[5]
        start = time.time()
        print(t1)
        print(t2)
        print(t3)
        print(t4)
        with open('logs/task_values.txt', 'w') as file:
            file.write(f"{t1}\n")
            file.write(f"{t2}\n")
            file.write(f"{t3}\n")
            file.write(f"{t4}\n")
        command1 = ["./" + str(t1), sys.argv[6], sys.argv[7], sys.argv[14]]
        command2 = ["./" + str(t2), sys.argv[8], sys.argv[9], sys.argv[14]]
        command3 = ["./" + str(t3), sys.argv[10], sys.argv[11], sys.argv[14]]
        command4 = ["./" + str(t4), sys.argv[12], sys.argv[13], sys.argv[14]]
        print(command1)
        print(command2)
        print(command3)
        print(command4)
        process_list.append(subprocess.Popen(command1))
        process_list.append(subprocess.Popen(command2))
        process_list.append(subprocess.Popen(command3))
        process_list.append(subprocess.Popen(command4))
        pid = process_list[0].pid
        pid2 = process_list[1].pid
        pid3 = process_list[2].pid
        pid4 = process_list[3].pid
        command5 = ["./server", str(len(process_list)), str(pid), str(pid2), str(pid3), str(pid4),sys.argv[6], sys.argv[8],sys.argv[10],sys.argv[12]]
        process_list.append(subprocess.Popen(command5))
        print(command5)
        timeout = float(sys.argv[14]) + 5
        while process_list:
            for process in process_list[:]:
                if process.poll() is not None:
                    process_list.remove(process)
                else:
                    elapsed_time = time.time() - start
                    if elapsed_time >= timeout:
                        process.kill()
                        process_list.remove(process)
                    else:
                        time.sleep(0.1)
