# FC-GPU

## Abstract

GPUs have recently been adopted in many real-time embedded systems. However, existing GPU scheduling
solutions are mostly open-loop and rely on the estimation of worst-case execution time (WCET). Although
adaptive solutions, such as feedback control scheduling, have been previously proposed to handle this challenge
for CPU-based real-time tasks, they cannot be directly applied to GPU, because GPUs have different and more
complex architectures and so schedulable utilization bounds cannot apply to GPUs yet. In this paper, we
propose FC-GPU, the first Feedback Control GPU scheduling framework for real-time embedded systems. To
model the GPU resource contention among tasks, we analytically derive a multi-input-multi-output (MIMO)
system model that captures the impacts of task rate adaptation on the response times of different tasks.
Building on this model, we design a MIMO controller that dynamically adjusts task rates based on measured
response times. Our extensive hardware testbed results on an Nvidia RTX 3090 GPU and an AMD MI-100
GPU demonstrate that FC-GPU can provide better real-time performance even when the task execution times
significantly increase at runtime.

## Note

FC-GPU is implemented as a user-space library. It can be used with any NVIDIA or AMD GPU, but control tuning must be done to enable FC-GPU to choose the appropriate coefficients. Currently, the coefficients are tuned offline and selected based on performance. This tuning can be adapted to other environments.

## Python Requirements

A file named `requirements.txt` is included and contains all Python packages used in our experiments. You can install them with:

```
pip install -r requirements.txt
```
