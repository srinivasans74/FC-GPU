import sys
import matplotlib.pyplot as plt


def read_times(filename):
    with open(filename, 'r') as f:
        return [float(line.strip()) for line in f if line.strip()]

def calculate_mean(values):
    return sum(values) / len(values) if values else 0.0

def plot_data(data_list, setpoints, labels, output_file):
    for i, data in enumerate(data_list):
        plt.plot(data, label=f'{labels[i]} (Setpoint={setpoints[i]})')
    plt.xlabel('Control Periods')
    plt.ylabel('Response Time Ratio (RTR)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python script.py <setpoint1> <setpoint2>")
        sys.exit(1)

    t1, t2 = sys.argv[1], sys.argv[2]

    reltime1 = read_times('logs/s1.txt')
    reltime2 = read_times('logs/s2.txt')

    sep1 = float(t1)
    sep2 = float(t2)

    mean_reltime1 = calculate_mean(reltime1)
    mean_reltime2 = calculate_mean(reltime2)

    with open('result.txt', 'a') as file:
        file.write(f"{t1}  {t2}\n")
        file.write(f"Mean reltime1: {mean_reltime1}\n")
        file.write(f"Mean reltime2: {mean_reltime2}\n\n")

    plot_data(
        [reltime1, reltime2],
        [sep1, sep2],
        ['t1', 't2'],
        f'figures/rtr_s1_{sep1}_{sep2}.pdf'
    )