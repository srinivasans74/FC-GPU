# Real-Time GPU Task Management with Dynamic Task Arrivals

This experiment demonstrates a real-time system designed to handle dynamic task arrivals on a GPU, specifically focusing on scenarios where new tasks, such as those arising from unexpected environmental events (e.g., an obstacle in an autonomous vehicle), necessitate rapid controller adjustments. Our approach, utilizing an FC-GPU (Flexible Control for GPU) framework, enables the system to dynamically switch between pre-tuned controller models to maintain system stability and prevent deadline misses during changes in task workload.

## Online Experiments: Dynamic Arrival of Tasks

In real-time systems, new tasks can originate from various sources, including user applications, system events, or processing sensor inputs. A prime example is in autonomous vehicles: if an unexpected obstacle, like a traffic cone, appears on the road, the vehicle must react swiftly to prevent a collision. This often involves promptly scheduling a new task to address the obstacle and ensure safe navigation. When such a situation occurs, the system's controller needs to adjust the rates of all active tasks within a small number of control periods to avoid significant deadline misses.

Our FC-GPU framework addresses this challenge by incorporating multiple controller models, each designed for a different number of concurrent tasks. At runtime, the system can seamlessly switch between these pre-validated controller models as tasks dynamically arrive or depart, ensuring continuous real-time performance and stability.

## Running the Experiment

To execute the dynamic task arrival experiment, use the provided shell script:

```bash
./experiment.sh
```

This script will compile and run the necessary components of the experiment, simulating dynamic task arrivals and the controller's response.

## Analyzing the Results

After running the experiment, you can visualize the system's performance and the controller's adjustments by opening the `plot.ipynb` Jupyter Notebook:

```bash
jupyter notebook plot.ipynb
```

This notebook contains code to generate figures illustrating various aspects of the experiment, such as task response times, task period adjustments.

## Important Note on GPU Tuning

**If you are using a GPU different from the one used for initial development and tuning, the controller gains (`a11` through `a44`) will likely need to be re-tuned.** These gains are critical for the stability of the control system and are highly dependent on the underlying hardware's performance characteristics.

We recommend performing a stability analysis (as per your system design) to determine the optimal gain values for your specific GPU to ensure the controller operates effectively and maintains real-time guarantees. Failure to re-tune these gains on different hardware may lead to performance degradation or instability.
