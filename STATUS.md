# Artifact Evaluation – STATUS

## Applied Badges

We are applying for the following artifact evaluation badges:

- Artifacts Available  
- Artifacts Evaluated – Functional  
- Artifacts Evaluated – Reusable

---

## Artifacts Available

The artifact has been open-sourced and is publicly available on GitHub:

- Repository: [https://github.com/srinivasans74/FC-GPU](https://github.com/srinivasans74/FC-GPU)
- License: MIT License (see `LICENSE` file in the repository)

We believe this artifact qualifies for the **Artifacts Available** badge because:
- The code is fully open-source and accessible to anyone.
- The repository includes the full implementation of FC-GPU and documentation for external use.
- Interested users can inspect, reuse, or build upon the provided feedback scheduling framework.

---

## Artifacts Evaluated – Functional

This artifact qualifies as **Functional** because all software dependencies and installation steps are clearly documented and reproducible:

- System requirements (e.g., OS, compilers, GPU toolchains) are listed in `REQUIREMENTS.md`.
- Python dependencies are pinned in `requirements.txt` for reproducibility.
- ROCm (for AMD) and CUDA/NVIDIA support are outlined.
- The `INSTALL.md` file describes how to install and verify setup.
- The software is verified to work on AMD Instinct MI100 and NVIDIA RTX 3090 GPUs.
Anyone following the installation guide on compatible hardware will be able to run FC-GPU as intended.

---

## Artifacts Evaluated – Reusable

We believe this artifact merits the **Artifacts Evaluated – Reusable** badge because:

- All code and instructions are modular, documented, and ready to extend.
- `README.md` includes instructions on how to execute the software and reproduce the results from the paper.
- Scripts for performance evaluation and plot generation are included.
- The system is designed to be extensible to other GPUs  with minimal modification.

Overall, this artifact can be repurposed, extended, or reused by researchers exploring real-time GPU scheduling.