# Explainable AI for Image Classification: Revealing and Interpreting Model Memorization

Master's thesis project, SRH University Heidelberg (Applied Computer Science, AI specialization).

This repository contains the LaTeX source of the thesis and the Python code used for the experiments. The experiments study how a convolutional network memorizes training data when labels or inputs are corrupted, and how explainability methods (Grad-CAM, Integrated Gradients) and adversarial attacks (FGSM, PGD) behave on such models.

> **Status: work in progress.** The thesis text is partly written, and several modules are still empty stubs. See [Project status](#project-status).

## Research question

Can explainability methods reveal when an image classifier has memorized its training data instead of learning generalizable features, and what do the attributions look like in that case?

## Experiments

The experiments follow Zhang et al. (2021), *Understanding deep learning (still) requires rethinking generalization*:

- **True labels:** the original CIFAR-10 training set.
- **Random labels:** training labels are shuffled, so the model can only fit the training set by memorizing it.
- **Random pixels:** each image is replaced by Gaussian noise with a fixed permutation, so there is no learnable signal.
- **Label corruption:** the fraction of corrupted labels is varied (0%, 50%, 100%) and time-to-overfit and test error are compared for a simple CNN and a ResNet-style model.

Dataset: CIFAR-10, with standard normalization and augmentation for the clean training runs.

## Repository structure

```
main.tex                      Thesis entry point (LaTeX)
chapters/                     Thesis chapters (introduction, background, methodology, implementation, evaluation, conclusion)
subsubsections/               Detailed background sections (memorization, attribution, white-box attacks, setup)
bibliography.bib              References
master_thesis/
  data/                       CIFAR-10 loader and the label/pixel randomization dataset (ZhangExperiment)
  models/                     SimpleCNN
  xai/                        Grad-CAM and Integrated Gradients (in progress)
  attacks/                    FGSM and PGD (in progress)
  memorization/               Memorization score (in progress)
  evaluation/                 Plotting utilities (metrics in progress)
  main.ipynb                  Notebook that runs the experiments end to end
```

## Requirements

- Python 3.11
- PyTorch 2.10, Torchvision 0.25
- Captum 0.8 (Integrated Gradients)
- pytorch-grad-cam (Grad-CAM)
- NumPy 1.26, Matplotlib 3.10, scikit-learn 1.8
- Jupyter / ipykernel

Install:

```bash
pip install torch torchvision matplotlib numpy scikit-learn captum grad-cam ipykernel
```

Training uses the MPS backend on Apple silicon when available and falls back to CPU otherwise.

## Data

CIFAR-10 is downloaded by the loader on first use. If you prefer to download it manually, place the extracted `cifar-10-batches-py/` folder under `master_thesis/data/`. The archive itself is git-ignored.

## Running the experiments

1. Open `master_thesis/main.ipynb` in Jupyter.
2. Update the `sys.path.append(...)` line so it points to your local `master_thesis/master_thesis` folder.
3. Run the cells in order. Training configuration (epochs, learning rate) is set in the training cells.

## Building the thesis

`main.tex` uses `fontspec`, so compile it with XeLaTeX or LuaLaTeX:

```bash
xelatex main.tex
biber main   # or bibtex, depending on the bibliography setup
xelatex main.tex
xelatex main.tex
```

## Project status

- Done: thesis introduction, background on neural network training, memorization, and white-box attacks.
- Done: data pipeline (CIFAR-10 loader, label and pixel randomization), SimpleCNN, training and evaluation loop in the notebook.
- In progress: implementation chapter (sections on model architecture, memorization analysis, XAI, and adversarial attacks are still empty), evaluation and results, conclusion, appendix.
- Not yet working: `attacks/pgd.py` (undefined variable and a typo in the first lines; needs fixing before use), empty modules `xai/gradcam.py`, `xai/integrated_gradients.py`, `memorization/memorization_score.py`, `evaluation/metrics.py`.

## License

All rights reserved. This thesis and its source code are shared for reference. Please contact the author before reusing the text or code.

## Author

Zeynep Öykü Erdem
