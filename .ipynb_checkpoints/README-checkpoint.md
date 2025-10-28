# Hyperparameters in Continual Learning: a Reality Check

[![Paper TMLR](https://img.shields.io/badge/Paper-TMLR-b31b1b.svg)](https://openreview.net/forum?id=hiiRCXmbAz) [![arXiv](https://img.shields.io/badge/arXiv-TBD-b31b1b.svg)](https://arxiv.org/abs/2403.09066) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation for the paper: **"Hyperparameters in Continual Learning: a Reality Check"** (TMLR 2025).

Authors: Sungmin Cha, Kyunghyun Cho

---

### ‼️ A Critical Note on Evaluation in Continual Learning

The primary goal of this repository is to **demonstrate a concrete implementation** of our proposed **Generalizable Two-phase Evaluation Protocol (GTEP)**. While this code allows for the reproduction of our experimental results, its ultimate purpose is to serve as a practical guide for the CL community.

#### Figure 1: The Flawed Conventional Protocol
![Conventional Protocol](figs/conventional.png)
> The conventional protocol tunes and evaluates on the same scenario, leading to overestimated performance.

We highlight a critical flaw in the conventional evaluation protocol, which is dominantly adopted in CL. This protocol tunes and evaluates an algorithm within the **same** CL scenario. As illustrated in Figure 1, this method is unrealistic for real-world applications (where future task data is not accessible) and leads to a significant **overestimation** of an algorithm's true CL capacity.

### The Generalizable Two-phase Evaluation Protocol (GTEP)

#### Figure 2: The Proposed GTEP Protocol
![GTEP Protocol](figs/gtep.png)
> GTEP separates tuning ($D^{HT}$) and evaluation ($D^{E}$) to measure true generalizability.

GTEP assesses an algorithm's generalizability by separating the evaluation process into two distinct phases:

1.  **Phase 1: Hyperparameter Tuning:** Identify the best hyperparameters ($\mathcal{H}^{*}$) in a *seen* scenario, which is generated from a hyperparameter-tuning dataset ($D^{HT}$).
2.  **Phase 2: Evaluation:** Apply these fixed, optimal hyperparameters ($\mathcal{H}^{*}$) to evaluate the algorithm's performance on a separate, *unseen* scenario generated from a different evaluation dataset ($D^{E}$).

The core idea is to measure if hyperparameters tuned on seen scenarios (i.e., simulated scenarios) can generalize to unseen scenarios (i.e., real-world scenarios), which is essential for any practical CL system.

By providing this reference implementation, we strongly encourage the CL community to adopt GTEP-based evaluation. We hope to see this protocol applied across diverse CL domains to ensure a more rigorous and realistic assessment of CL algorithms' **generalizability**.


---

### About This Repository

This codebase is an implementation of GTEP built upon two excellent, widely-used CL codebases:

* **[PyCIL](https://github.com/G-U-N/PyCIL):** Used for experiments *without* pretrained models.
* **[LAMBDA-PILOT](https://github.com/LAMDA-CL/LAMDA-PILOT):** Used for experiments *with* pretrained models.

We have integrated the GTEP workflow (i.e., `main_search.sh` and `main_test.sh` scripts) into these frameworks.

### Setup and Installation

#### 1. Environment
Please follow the recommended environment setup from the original repositories ([PyCIL](https://github.com/G-U-N/PyCIL) and [LAMBDA-PILOT](https://github.com/LAMDA-CL/LAMDA-PILOT)).

#### 2. Datasets
You must download the following datasets used in our experiments:
* [ImageNet100](https://www.kaggle.com/datasets/ambityga/imagenet100)
* [CUB-200](https://www.vision.caltech.edu/datasets/cub_200_2011/)
* [ImageNet-A](https://github.com/hendrycks/natural-adv-examples)
* [ImageNet-R](https://github.com/hendrycks/imagenet-r)

#### 3. Dataset Configuration
**This is a critical step.** After downloading the datasets:

1.  **Split the datasets:** You must manually split the datasets to create the disjoint scenarios for the $D^{HT}$ (tuning) and $D^{E}$ (evaluation) phases. (For complete details on how we created our splits, please refer to Section 4.1 and 4.2 of our paper).
2.  **Update dataset path:** Open `utils/data.py` in the respective codebase.
3.  **Set paths:** Update the `train_dir` and `test_dir` variables to point to the correct file paths for your local dataset locations.

### Running Experiments with GTEP

The experimental workflow is divided into GTEP's two phases.

#### Phase 1: Hyperparameter Search (Tuning)
To run the hyperparameter search phase on the $D^{HT}$ dataset, execute:
```bash
bash main_search.sh
````

This will iterate through various hyperparameter configurations defined in the script. The search ranges are based on the values reported in the original papers for each algorithm.

#### Phase 2: Evaluation

After identifying the best hyperparameters from Phase 1, update the test script with them. Then, run the evaluation phase on the $D^{E}$ dataset:

```bash
bash main_test.sh
```

This script will run the evaluation using the *single set* of best hyperparameters to produce the final performance, which measures the algorithm's generalizability.

### Hyperparameters

  * The test code (`main_test.sh`) provided in this repository is pre-configured with the best hyperparameters found for the **ImageNet-100** dataset.
  * The best hyperparameters for **all other datasets** and scenarios are reported in the **Appendix** of our paper. You will need to manually update the `main_test.sh` script with these values to reproduce those results.

### Citation

If you find this work or code useful for your research, please cite our paper:

```bibtex
@article{cha2025hyperparameters,
  title={Hyperparameters in Continual Learning: A Reality Check},
  author={Sungmin Cha and Kyunghyun Cho},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2025},
  url={https://openreview.net/forum?id=hiiRCXmbAz},
  note={}
}
```

### Acknowledgements

This code was built upon the [PyCIL](https://github.com/G-U-N/PyCIL) and [LAMBDA-PILOT](https://github.com/LAMDA-CL/LAMDA-PILOT) repositories. We express our sincere gratitude to the original authors for their valuable contributions and for making their code publicly available!!
