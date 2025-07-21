import os
import subprocess
import time
import sys
from time import sleep

if __name__ == '__main__':
    process_list = []

    if len(sys.argv) < 2:
        print("Usage: script.py <mode> [additional arguments]")
        sys.exit(1)

    mode = int(sys.argv[1])
    print(f"Mode: {mode}")

    if mode == 1:
        if len(sys.argv) < 5:
            print("Usage for mode 1: script.py 1 <t1> <period> <termination>")
            sys.exit(1)

        command1 = ["./t1", sys.argv[3], sys.argv[4], sys.argv[5]]
        print(f"Running command: {command1}")
        process_list.append(subprocess.Popen(command1))

        # Record the start time
        start = time.time()

        # Timeout is calculated as the termination time plus a buffer (5 seconds)
        timeout = float(sys.argv[5]) + 5

        while process_list:
            for process in process_list[:]:
                if process.poll() is not None:
                    process_list.remove(process)
                else:
                    elapsed_time = time.time() - start
                    if elapsed_time >= timeout:
                        print(f"Timeout reached. Killing process {process.pid}")
                        process.kill()
                        process_list.remove(process)
                    else:
                        time.sleep(0.1)

    elif mode == 4:
        if len(sys.argv) < 15:
            print("Usage for mode 4: script.py 4 <t1> <t2> <t3> <t4> <arg1> <arg2> <arg3> <arg4> <arg5> <arg6> <arg7> <arg8> <termination>")
            sys.exit(1)

        t1 = sys.argv[2]
        t2 = sys.argv[3]
        t3 = sys.argv[4]
        t4 = sys.argv[5]
        print(f"Task binaries: {t1}, {t2}, {t3}, {t4}")

        # Record the start time
        start = time.time()

        with open('logs/task_values.txt', 'w') as file:
            file.write(f"{t1}\n")
            file.write(f"{t2}\n")
            file.write(f"{t3}\n")
            file.write(f"{t4}\n")

        command1 = ["./" + t1, sys.argv[6], sys.argv[7], sys.argv[14]]
        command2 = ["./" + t2, sys.argv[8], sys.argv[9], sys.argv[14]]
        command3 = ["./" + t3, sys.argv[10], sys.argv[11], sys.argv[14]]
        command4 = ["./" + t4, sys.argv[12], sys.argv[13], sys.argv[14]]

        print(f"Running command: {command1}")
        print(f"Running command: {command2}")
        print(f"Running command: {command3}")
        print(f"Running command: {command4}")

        process_list.append(subprocess.Popen(command1))
        process_list.append(subprocess.Popen(command2))
        process_list.append(subprocess.Popen(command3))
        process_list.append(subprocess.Popen(command4))

        # Timeout is calculated as the termination time plus a buffer (5 seconds)
        timeout = float(sys.argv[14]) + 5

        while process_list:
            for process in process_list[:]:
                if process.poll() is not None:
                    process_list.remove(process)
                else:
                    elapsed_time = time.time() - start
                    if elapsed_time >= timeout:
                        print(f"Timeout reached. Killing process {process.pid}")
                        process.kill()
                        process_list.remove(process)
                    else:
                        time.sleep(0.1)
    else:
        print("Invalid mode. Use 1 or 4.")
        sys.exit(1)
