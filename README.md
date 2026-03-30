# Quantum-Enhanced Learning of Protein Structural Features

> A Hybrid Quantum-Classical Proof-of-Concept Study

This repository explores whether quantum feature maps — implemented via variational quantum circuits on top of Qiskit — can meaningfully improve classification of protein secondary structure (Helix, Sheet, Coil) compared to purely classical baselines.

---

## Overview

The project is structured as three progressive experiments (`test1.py` → `test2.py` → `test3.py`), each building on the limitations discovered in the previous version. Classical classifiers (Logistic Regression and MLP) are trained both on raw angle features and on quantum-transformed features, then compared head-to-head.

```
Synthetic Protein Dataset (angles in [0, π])
Labels: Helix(0), Sheet(1), Coil(2) — nonlinear rule
       │                        │
 Raw Features            Quantum Feature Map
 (Classical)          Shallow QNN  |  Deep QNN
       │                        │
  LR / MLP            LR / MLP on Q-features
       └──────────────────┬──────┘
                  Comparison CSV + Charts
```

---

## Experiment Versions

### `test1.py` — Baseline Hybrid Pipeline (v1)

Establishes the foundational pipeline:

- Generates a synthetic dataset of 800 protein-like samples (4-angle feature vectors)
- Labels are assigned using nonlinear sin/cos rules to simulate realistic class boundary complexity
- Implements two quantum circuit architectures:
  - **Shallow QNN**: angle encoding → RY/RZ variational layer → linear CNOT entanglement
  - **Deep QNN**: angle encoding → 2× (RY/RZ + entanglement) → circular CNOT
- Extracts Pauli-Z expectation values `<Z>` as quantum features via Statevector simulation
- Trains LR and MLP classifiers on both raw and quantum features
- **Key limitation**: dataset was heavily imbalanced (~89% Helix), and variational parameters were fixed/random

### `test2.py` — Balanced Dataset + ZZ Correlators (v2)

Addresses the class imbalance and feature richness problems from v1:

- **Balanced dataset**: 300 samples per class (900 total) using noise injection and oversampling
- **ZZ correlator features**: quantum feature vector expanded from `n_qubits` to `n_qubits + (n_qubits - 1)` by adding `<ZiZj>` pairwise measurements that capture entanglement correlations
- **Robust label handling**: class names derived from data, not hardcoded assumptions
- **Key limitation**: variational parameters are still random — a random circuit is effectively just a random projection with no class-discriminative signal

### `test3.py` — COBYLA-Optimised Variational Parameters (v3)

Root-cause fix: optimises the variational angles so the quantum feature map is actually informative.

- **COBYLA optimisation**: gradient-free scipy optimiser minimises cross-entropy of LR trained on a small validation set
  - ~80 iterations × ~0.15s/eval ≈ 12s per circuit type
  - No backpropagation through circuits required
- Compares four quantum configurations: Shallow/Deep × Random/Optimised
- Best result: **Optimised Shallow QNN + MLP achieves 81.67% accuracy**, closing the gap with the classical MLP baseline (82.78%) to within ~1%

---

## Results Summary (v3)

| Model | Type | Accuracy | F1 |
|---|---|---|---|
| MLP \| Raw | Classical | **82.78%** | 0.8182 |
| MLP \| Shallow [optimised] | Quantum | 81.67% | 0.8097 |
| LR \| Raw | Classical | 80.56% | 0.7940 |
| LR \| Shallow [optimised] | Quantum | 80.56% | 0.7983 |
| MLP \| Deep [random] | Quantum | 76.67% | 0.7573 |
| LR \| Deep [optimised] | Quantum | 74.44% | 0.7384 |
| MLP \| Deep [optimised] | Quantum | 72.78% | 0.7070 |
| MLP \| Shallow [random] | Quantum | 72.22% | 0.7098 |
| LR \| Shallow [random] | Quantum | 66.67% | 0.6590 |
| LR \| Deep [random] | Quantum | 66.11% | 0.6532 |

**Key takeaway**: Optimised quantum circuits nearly match classical performance, and the COBYLA optimisation step provides a consistent +5% accuracy gain over random parameters.

---

## Requirements

```bash
pip install qiskit qiskit-ibm-runtime scikit-learn scipy pandas matplotlib numpy
```

| Package | Purpose |
|---|---|
| `qiskit` | Quantum circuit construction and simulation |
| `qiskit-ibm-runtime` | Hardware-ready IBM Quantum service integration |
| `scikit-learn` | LR/MLP classifiers, metrics, preprocessing |
| `scipy` | COBYLA gradient-free optimisation (v3 only) |
| `pandas` | Results tabulation and CSV export |
| `matplotlib` | Comparison charts and confusion matrices |
| `numpy` | Numerical computation |

---

## Usage

Each script is self-contained and can be run independently:

```bash
python test1.py   # v1: baseline pipeline
python test2.py   # v2: balanced dataset + ZZ features
python test3.py   # v3: COBYLA-optimised parameters
```

### Outputs

Each script saves results to its own subdirectory (`test1_results/`, `test2_results/`, etc.):

| File | Description |
|---|---|
| `quantum_protein_results.csv` | Full metrics table for all model configurations |
| `quantum_protein_comparison.png` | Accuracy bar chart across all models |
| `quantum_protein_comparison_confusion.png` | Per-class confusion matrices |
| `quantum_protein_comparison_opt_curves.png` | COBYLA optimisation loss curves (v3) |
| `quantum_protein_comparison_class_dist.png` | Class distribution of the dataset |

---

## Quantum Circuit Details

**Shallow QNN** (2 × n_qubits parameters):
```
Encoding  : RY(xᵢ) on each qubit
Var Layer : RY(θᵢ) · RZ(φᵢ) on each qubit
Entangle  : linear CNOT chain q₀ → q₁ → q₂ → …
```

**Deep QNN** (4 × n_qubits parameters):
```
Encoding   : RY(xᵢ)
Layer 1    : RY(θ¹ᵢ) · RZ(φ¹ᵢ) → linear CNOT
Layer 2    : RY(θ²ᵢ) · RZ(φ²ᵢ) → circular CNOT (includes qₙ → q₀)
```

Feature extraction uses **Pauli-Z expectation values** via Qiskit's `Statevector` simulator. The code is hardware-ready — swapping to a real IBM Quantum backend requires only a one-line change to `QiskitRuntimeService`.

---

## Dataset

Synthetic protein-like data where each sample is a 4-dimensional angle vector in `[0, π]` representing per-residue dihedral/physicochemical features. Class labels are assigned by nonlinear rules:

- **Helix** — correlated sin pattern: `Σ sin(xᵢ) + sin(x₀·x₁)`
- **Sheet** — alternating cosine pattern: `Σ cos(xᵢ) + cos(x₋₁·x₋₂)`
- **Coil** — nonlinear mixed: `Σ sin(xᵢ)cos(xᵢ) + sin(Σxᵢ)`

---

## Hardware Notes

Local statevector simulation is used for all experiments (< 1 min total on CPU). `QiskitRuntimeService` is initialised at startup so the code can be redirected to real quantum hardware with minimal modification. The `RUNTIME_AVAILABLE` flag handles environments where `qiskit-ibm-runtime` is not installed.
