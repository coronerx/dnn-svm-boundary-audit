# Classifier Boundary Audit: Investigating Interpolation Vulnerabilities

A comprehensive diagnostic toolkit designed to evaluate and audit the structural robustness of Deep Neural Networks (ResNet) and Support Vector Machines (SVM). This project simulates input perturbations via pixel-level linear interpolation (image blending) to detect latent manifold shifts, decision boundary vulnerabilities, and predictive hallucinations.

## Features
- **Hybrid Architecture Evaluation**: Comparative vulnerability analysis between a deep convolutional framework (ResNet/LeNet-5 hybrid) and a classic statistical boundary classifier (RBF-Kernel SVM).
- **Latent Manifold Mapping**: Extracting 128-dimensional deep feature representations to analyze classifier trajectories during continuous state-space transitions.
- **Automated Pipeline**: End-to-end data acquisition and ingestion pipeline capable of parsing high-throughput multi-stage input frames.
- **Vulnerability Diagnostics**: Detection of localized "hallucinations" where linear pixel combinations trick non-linear decision boundaries into predicting out-of-distribution (OOD) classes.

---

## Analysis

When evaluating image transitions (e.g., smoothly interpolating an image from digit `0` to digit `4`), the classifiers exhibit distinct boundary behaviors:

1. **Feature Space vs. Pixel Space**: Linear blending in pixel space results in non-linear trajectories within the learned latent space, potentially driving intermediate representations into unexpected classification regions.
2. **Boundary Vulnerability (Hallucination)**: During specific topological morphing (such as `0` to `4` or structural variations of `1`), overlaying pixel structures can geometrically mimic alternative digit manifolds (e.g., class `8`), causing both models to output high-confidence false-positives at intermediate steps.
3. **Out-of-Distribution (OOD) Sensitivity**: Testing on un-regularized custom handwritten inputs revealed systemic vulnerabilities to minor covariate shifts (e.g., misclassifying a digit `7` lacking a crossbar due to distribution shifts).

---

## Project Structure

```text
├── src/
│   ├── main_audit.py        # Core engine running the ResNet-SVM inference and plotting
│   └── utils_prepare.py     # Image preprocessing, inversion, and normalization utils
├── docs/
│   └── answers.md           # Theoretical analysis and research question write-up
└── requirements.txt         # Dependency configurations
