# FC-GPU

## Abstract

GPUs have recently been adopted in many real-time embedded systems. However, existing GPU scheduling solutions are mostly open-loop and rely on the estimation of worst-case execution time (WCET). Although adaptive solutions, such as feedback control scheduling, have been previously proposed to handle this challenge for CPU-based real-time tasks, they cannot be directly applied to GPUs, because GPUs have different and more complex architectures and schedulable utilization bounds do not yet apply to them.
In this paper, we propose **FC-GPU**, the first **Feedback Control GPU scheduling framework** for real-time embedded systems. To model the GPU resource contention among tasks, we analytically derive a **multi-input-multi-output (MIMO)** system model that captures the impacts of task rate adaptation on the response times of different tasks. Building on this model, we design a MIMO controller that dynamically adjusts task rates based on measured response times.
Our extensive hardware testbed results on an Nvidia RTX 3090 GPU and an AMD MI-100 GPU demonstrate that FC-GPU can provide better real-time performance, even when task execution times significantly increase at runtime.


## Python Requirements

All Python dependencies used in the experiments are listed in [`requirements.txt`](./requirements.txt). You can install them using:

```bash
pip install -r requirements.txt


Please go through the ['REQUIREMENTS.md'](./REQUIREMENTS.md) since somtimes we need to adjust the gain parameteres.
```
---

## Experiments

To run the experiments, please refer to [`INSTALL.md`](./INSTALL.md), which contains complete setup and usage instructions.

All experimental results, including overhead measurements, are available in this repository. Each experiment includes fully-commented code and plotting scripts. Various workloads can be tested using the provided configurations.

---

## Citation

If you use this code or dataset in your work, please cite the following paper:

```bibtex
@article{subramaniyan2025fcgpu,
  author    = {Srinivasan Subramaniyan and Xiaorui Wang},
  title     = {{FC-GPU: Feedback Control GPU Scheduling for Real-time Embedded Systems}},
  journal   = {ACM Transactions on Embedded Computing Systems (TECS)},
  year      = {2025},
  month     = sep,
  publisher = {ACM},
  note      = {To appear. Presented at EMSOFT 2025, Taipei, Taiwan}
}
```

---

## Repository Structure

Other documentation files available in this repository:

- [`INSTALL.md`](./INSTALL.md) – Step-by-step installation guide  
- [`REQUIREMENTS.md`](./REQUIREMENTS.md) – Additional dependencies and hardware notes  
- [`STATUS.md`](./STATUS.md) – Status updates and to-dos  
- [`LICENSE`](./LICENSE) – Licensing information  

---

For more details, please explore the individual documents listed above.
