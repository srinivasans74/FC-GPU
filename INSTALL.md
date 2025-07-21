# INSTALL

## Overview

This repository contains code to reproduce key figures from the associated research paper:

- **Overhead/**: Experiments for overhead measurement — reproduces **Figure 5**
- **Modelling/**: Code to reproduce **Figure 3**
- **baseline/**: Scripts to reproduce all baseline figures

---

## Installation Instructions

### 1. Set Up Environment

It is recommended to use a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
```

### 2. Install Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

> Alternatively, see `REQUIREMENTS.md` for detailed descriptions of dependencies.

---

## Basic Usage & Verification

### Run Overhead Measurement (Reproduces Figure 5)

```bash
cd Overhead/
./experiment.sh
```

Expected output:
```
Running overhead experiments...
Figure 5 saved as overhead_figure5.png
```

---

### Run Modeling Script (Reproduces Figure 3)

```bash
cd ../Modelling/
./run_model.sh
```

Expected output:
```
Generating model analysis...
Figure 3 saved as model_figure3.png
```

---

### Experiments

---
> All the experiments are in the `Experiments/`

---

## Notes

- All scripts save their figures in their respective folders.
- Ensure any `.sh` scripts are executable: `chmod +x scriptname.sh`
- For any additional setup or dataset paths, refer to `README.md` or in-script comments.

---

## Help

Please open an issue if you encounter problems or contact the maintainer listed in the `LICENSE` file.
