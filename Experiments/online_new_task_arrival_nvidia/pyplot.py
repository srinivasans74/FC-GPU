import numpy as np
import matplotlib.pyplot as plt
import sys

def read_times(file_path):
    with open(file_path, 'r') as f:
        return [float(line.strip()) for line in f.readlines()]

def plot_data(reltimes, sep_values, labels, save_path):
    fig, ax = plt.subplots(figsize=(3, 3))

    for i, reltime in enumerate(reltimes):
        plt.plot(reltime[10:-3], '-', label=labels[i])
        sep = [sep_values[i] for _ in range(len(reltime))]
        plt.plot(sep[10:-3], '--', label=f's{i+1}')
    
    plt.ylabel('RTR')
    plt.xlabel('Control Period')
    fig.tight_layout()
    plt.legend(ncols=2)
    plt.grid()
    plt.ylim([0,2])
    plt.savefig(save_path)

if __name__ == '__main__':
    if len(sys.argv) == 3:
        t1 = sys.argv[1]
        t2 = sys.argv[2]

        reltime1 = read_times('logs/s11.txt')
        reltime2 = read_times('logs/s22.txt')

        sep1 = float(t1)
        sep2 = float(t2)

        plot_data(
            [reltime1, reltime2],
            [sep1, sep2],
            ['t1', 't2'],
            f'figures/rtr_s1_{sep1}_{sep2}.pdf'
        )
    elif len(sys.argv) == 4:
        t1 = sys.argv[1]
        t2 = sys.argv[2]
        t3 = sys.argv[3]

        reltime1 = read_times('logs/s11.txt')
        reltime2 = read_times('logs/s22.txt')
        reltime3 = read_times('logs/s33.txt')

        sep1 = float(t1)
        sep2 = float(t2)
        sep3 = float(t3)

        plot_data(
            [reltime1, reltime2, reltime3],
            [sep1, sep2, sep3],
            ['t1', 't2', 't3'],
            f'figures/rtr_s1_{sep1}_{sep2}_{sep3}.pdf'
        )
 

    elif len(sys.argv) == 5:
        t1 = sys.argv[1]
        t2 = sys.argv[2]
        t3 = sys.argv[3]
        t4 = sys.argv[4]

        reltime1 = read_times('logs/s11.txt')
        reltime2 = read_times('logs/s22.txt')
        reltime3 = read_times('logs/s33.txt')
        reltime4 = read_times('logs/s44.txt')

        sep1 = float(t1)
        sep2 = float(t2)
        sep3 = float(t3)
        sep4 = float(t4)

        plot_data(
            [reltime1, reltime2, reltime3,reltime4],
            [sep1, sep2, sep3,sep4],
            ['t1', 't2', 't3','t4'],
            f'figures/rtr_s1_{sep1}_{sep2}_{sep3}_{sep4}.pdf'
        )
    else:
        print("Invalid number of arguments. Provide 2 or 3 arguments.")