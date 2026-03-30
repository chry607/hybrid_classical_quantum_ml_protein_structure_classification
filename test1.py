"""
==============================================================================
  Quantum-Enhanced Learning of Protein Structural Features at Small Scale
  A Hybrid Quantum-Classical Proof-of-Concept Study
==============================================================================

Architecture Overview:
  ┌─────────────────────────────────────────────────────────┐
  │  Synthetic Protein Dataset (angles in [0, π])           │
  │  Labels: Helix(0), Sheet(1), Coil(2)  — nonlinear rule  │
  └────────────┬────────────────────────┬────────────────────┘
               │                        │
       ┌───────▼──────┐        ┌────────▼───────────────┐
       │  Raw Features│        │  Quantum Feature Map   │
       │  (Classical) │        │  Shallow / Deep QNN    │
       └───────┬──────┘        └────────┬───────────────┘
               │                        │
       ┌───────▼──────┐        ┌────────▼───────────────┐
       │  LR / MLP    │        │  LR / MLP on QFeatures │
       └───────┬──────┘        └────────┬───────────────┘
               └──────────┬─────────────┘
                  ┌────────▼────────┐
                  │  Comparison CSV │
                  │  + Bar Charts   │
                  └─────────────────┘

Quantum Circuit Strategy:
  - Shallow QNN  : encode → RY/RZ variational → linear CNOT
  - Deep   QNN   : encode → 2× (RY/RZ + entangle) → circular CNOT
  - Feature extraction: Pauli-Z expectation values via Statevector
  - QiskitRuntimeService is initialised (hardware-ready); local
    statevector simulation is used for speed (<1 min on CPU).

Requirements:
    pip install qiskit qiskit-ibm-runtime scikit-learn pandas matplotlib numpy
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
import time
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# Qiskit core
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Qiskit IBM Runtime — initialise the service for hardware-ready execution
try:
    from qiskit_ibm_runtime import QiskitRuntimeService

    RUNTIME_AVAILABLE = True
except ImportError:
    RUNTIME_AVAILABLE = False

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CFG = dict(
    n_samples=800,  # total dataset size
    seq_len=4,  # feature length == number of qubits
    test_size=0.20,  # 80/20 split
    random_seed=42,
    output_csv="quantum_protein_results.csv",
    output_png="quantum_protein_comparison.png",
    class_names=["Helix", "Sheet", "Coil"],
)
N_QUBITS = CFG["seq_len"]

# ==============================================================================
# 1.  DATASET GENERATION
# ==============================================================================


def generate_protein_dataset(n_samples: int, seq_len: int, seed: int):
    """
    Synthetic protein-like dataset.

    Each sample is a fixed-length vector of angles in [0, π] that
    loosely represent per-residue dihedral / physicochemical features.

    Label assignment uses nonlinear sin/cos rules to create curved,
    non-trivially separable class boundaries — mimicking the complexity
    of secondary structure prediction:

        Helix score  = Σ sin(xᵢ) + sin(x₀·x₁)          correlated
        Sheet score  = Σ cos(xᵢ) + cos(x₋₁·x₋₂)        alternating
        Coil  score  = Σ sin(xᵢ)cos(xᵢ) + sin(Σxᵢ)     mixed
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, np.pi, size=(n_samples, seq_len))

    s = np.zeros((n_samples, 3))
    # Helix: strongly correlated sin pattern
    s[:, 0] = (
        np.sum(np.sin(X), axis=1)
        + np.sin(X[:, 0] * X[:, 1])
        + 0.3 * np.sin(2 * X[:, 0])
    )
    # Sheet: cosine / alternating pattern
    s[:, 1] = (
        np.sum(np.cos(X), axis=1)
        + np.cos(X[:, -1] * X[:, -2])
        + 0.3 * np.cos(2 * X[:, -1])
    )
    # Coil: nonlinear mixed
    s[:, 2] = (
        np.sum(np.sin(X) * np.cos(X), axis=1)
        + np.sin(np.sum(X, axis=1))
        + 0.3 * np.cos(X[:, 0] - X[:, -1])
    )

    y = np.argmax(s, axis=1)
    return X, y


# ==============================================================================
# 2.  QUANTUM CIRCUITS
# ==============================================================================


def build_shallow_circuit(
    x: np.ndarray, n_qubits: int, params: np.ndarray
) -> QuantumCircuit:
    """
    Shallow variational circuit (1 variational layer):

        Encoding  : RY(xᵢ)  on each qubit
        Var Layer : RY(θᵢ) · RZ(φᵢ)  on each qubit
        Entangle  : linear CNOT chain  q₀→q₁→q₂→…

    params shape: (2 × n_qubits,)  [θ₀…θₙ, φ₀…φₙ]
    """
    qc = QuantumCircuit(n_qubits)

    # ── Angle encoding ─────────────────────────────────────────────────────
    for i in range(n_qubits):
        qc.ry(float(x[i]), i)

    # ── Variational layer ───────────────────────────────────────────────────
    for i in range(n_qubits):
        qc.ry(float(params[i]), i)  # rotation in Y
        qc.rz(float(params[n_qubits + i]), i)  # rotation in Z

    # ── Linear entanglement ─────────────────────────────────────────────────
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)

    return qc


def build_deep_circuit(
    x: np.ndarray, n_qubits: int, params: np.ndarray
) -> QuantumCircuit:
    """
    Deep variational circuit (2 variational layers + richer entanglement):

        Encoding   : RY(xᵢ)
        Layer 1    : RY(θ¹ᵢ) · RZ(φ¹ᵢ) → linear CNOT
        Layer 2    : RY(θ²ᵢ) · RZ(φ²ᵢ) → circular CNOT (includes qₙ→q₀)

    params shape: (4 × n_qubits,)
    """
    qc = QuantumCircuit(n_qubits)
    L = n_qubits  # params per sub-block

    # ── Angle encoding ─────────────────────────────────────────────────────
    for i in range(n_qubits):
        qc.ry(float(x[i]), i)

    # ── Variational layer 1 ─────────────────────────────────────────────────
    for i in range(n_qubits):
        qc.ry(float(params[i]), i)
        qc.rz(float(params[L + i]), i)
    for i in range(n_qubits - 1):  # linear CNOT
        qc.cx(i, i + 1)

    # ── Variational layer 2 ─────────────────────────────────────────────────
    for i in range(n_qubits):
        qc.ry(float(params[2 * L + i]), i)
        qc.rz(float(params[3 * L + i]), i)
    for i in range(n_qubits - 1):  # circular CNOT chain
        qc.cx(i, i + 1)
    qc.cx(n_qubits - 1, 0)  # close the ring

    return qc


# ==============================================================================
# 3.  QUANTUM FEATURE EXTRACTION
# ==============================================================================


def pauli_z_operators(n_qubits: int) -> list:
    """
    Build single-qubit Pauli-Z operators for every qubit.

    Qiskit's Pauli string convention is little-endian (rightmost char = qubit 0),
    so for qubit i out of n_qubits we place 'Z' at position i from the right:

        qubit 0  →  "III…Z"    (Z in LSB position)
        qubit 1  →  "II…ZI"
        …
    """
    ops = []
    for i in range(n_qubits):
        pauli_str = "I" * (n_qubits - 1 - i) + "Z" + "I" * i
        ops.append(SparsePauliOp(pauli_str))
    return ops


def extract_quantum_features(
    X: np.ndarray,
    circuit_fn,
    params: np.ndarray,
    n_qubits: int,
    label: str = "",
) -> np.ndarray:
    """
    For every sample in X:
      1. Build the parameterised quantum circuit
      2. Evolve the statevector
      3. Compute ⟨Z⟩ for each qubit  →  feature vector of length n_qubits

    This approach avoids backprop through the quantum layer; parameters
    are fixed random initialisations (a quantum feature-map / kernel view).
    """
    paulis = pauli_z_operators(n_qubits)
    features = np.zeros((len(X), n_qubits))

    for idx, x in enumerate(X):
        qc = circuit_fn(x, n_qubits, params)
        sv = Statevector.from_instruction(qc)
        for j, op in enumerate(paulis):
            features[idx, j] = float(sv.expectation_value(op).real)

    return features


# ==============================================================================
# 4.  CLASSICAL TRAINING HELPERS
# ==============================================================================


def make_scaler_and_fit(X_tr: np.ndarray, X_te: np.ndarray):
    """Standard-scale features; fit on train, apply to test."""
    sc = StandardScaler()
    return sc.fit_transform(X_tr), sc.transform(X_te), sc


def train_lr(X_tr, y_tr):
    """Logistic Regression with L-BFGS solver."""
    clf = LogisticRegression(
        max_iter=600, solver="lbfgs", random_state=CFG["random_seed"]
    )
    clf.fit(X_tr, y_tr)
    return clf


def train_mlp(X_tr, y_tr):
    """Small MLP: two hidden layers (64 → 32) with ReLU / adam."""
    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=600,
        random_state=CFG["random_seed"],
        early_stopping=True,
        n_iter_no_change=20,
    )
    clf.fit(X_tr, y_tr)
    return clf


# ==============================================================================
# 5.  EVALUATION HELPER
# ==============================================================================


def evaluate(
    clf, X_te: np.ndarray, y_te: np.ndarray, label: str, elapsed: float
) -> dict:
    """
    Compute accuracy, weighted precision / recall / F1.
    Prints a full sklearn classification report and returns a metrics dict.
    """
    y_pred = clf.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_te, y_pred, average="weighted", zero_division=0
    )

    bar = "─" * 56
    print(f"\n  ┌{bar}┐")
    print(f"  │  {label:<52s}  │")
    print(f"  ├{bar}┤")
    print(f"  │  Accuracy  : {acc:>7.4f}                                 │")
    print(f"  │  Precision : {prec:>7.4f}   Recall : {rec:>7.4f}            │")
    print(f"  │  F1-Score  : {f1:>7.4f}   Time   : {elapsed:>6.2f}s            │")
    print(f"  └{bar}┘")
    print()
    # Get unique labels from test and predictions to handle missing classes
    unique_labels = np.unique(np.concatenate([y_te, y_pred]))
    target_names = [CFG["class_names"][i] for i in unique_labels]
    print(
        classification_report(
            y_te,
            y_pred,
            labels=unique_labels,
            target_names=target_names,
            zero_division=0,
        )
    )

    return {
        "Model": label,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "Time_s": round(elapsed, 3),
    }


# ==============================================================================
# 6.  VISUALISATION
# ==============================================================================

PALETTE = ["#1565C0", "#2E7D32", "#6A1B9A", "#E65100", "#00695C", "#AD1457"]


def plot_results(df: pd.DataFrame, save_path: str):
    """
    Three-panel figure:
      (A) Accuracy bar chart
      (B) F1-Score bar chart
      (C) Timing bar chart (log scale)
    """
    labels = df["Model"].tolist()
    n = len(labels)
    colours = PALETTE[:n]
    x_pos = np.arange(n)
    w = 0.60

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(
        "Quantum-Enhanced Protein Structure Learning — Model Comparison",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )

    def _bar(ax, values, title, ylabel, ylim=(0, 1.15), fmt="{:.3f}"):
        bars = ax.bar(
            x_pos,
            values,
            width=w,
            color=colours,
            alpha=0.88,
            edgecolor="white",
            linewidth=1.2,
        )
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=9)
        ax.set_ylim(*ylim)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02 * ylim[1],
                fmt.format(val),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    _bar(axes[0], df["Accuracy"].tolist(), "Accuracy", "Accuracy")
    _bar(axes[1], df["F1"].tolist(), "F1-Score", "Weighted F1")
    _bar(
        axes[2],
        df["Time_s"].tolist(),
        "Wall-clock Time",
        "Seconds",
        ylim=(0, max(df["Time_s"]) * 1.4),
        fmt="{:.2f}s",
    )

    # Colour legend
    patches = [mpatches.Patch(color=colours[i], label=labels[i]) for i in range(n)]
    fig.legend(
        handles=patches,
        loc="lower center",
        ncol=n,
        bbox_to_anchor=(0.5, -0.12),
        fontsize=9,
        frameon=False,
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  [✓] Chart saved → {save_path}")


def plot_confusion_matrices(results_raw: list, save_path_prefix: str):
    """
    One confusion matrix per model — saved as a second figure.
    results_raw: list of (label, y_true, y_pred)
    """
    n = len(results_raw)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, (label, y_te, y_pred) in zip(axes, results_raw):
        cm = confusion_matrix(y_te, y_pred)
        # Get unique labels present in test and predictions
        unique_labels = np.unique(np.concatenate([y_te, y_pred]))
        target_names = [CFG["class_names"][i] for i in unique_labels]
        n_classes = len(unique_labels)

        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(target_names, rotation=30, fontsize=8)
        ax.set_yticklabels(target_names, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=11,
                    fontweight="bold",
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Confusion Matrices", fontsize=13, fontweight="bold")
    plt.tight_layout()
    conf_path = save_path_prefix.replace(".png", "_confusion.png")
    fig.savefig(conf_path, dpi=150, bbox_inches="tight")
    print(f"  [✓] Confusion matrices saved → {conf_path}")


# ==============================================================================
# 7.  MAIN PIPELINE
# ==============================================================================


def main():
    t_total_start = time.time()

    # ── Banner ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  Quantum-Enhanced Protein Structure Learning")
    print("  Hybrid QNN Proof-of-Concept Study")
    print("=" * 62)

    # ── QiskitRuntimeService ───────────────────────────────────────────────────
    print("\n[0]  Runtime Setup")
    if RUNTIME_AVAILABLE:
        try:
            service = QiskitRuntimeService()
            backends = service.backends()
            print(f"  [✓] QiskitRuntimeService connected.")
            print(f"      Available backends : {[b.name for b in backends[:4]]} …")
            print("      ℹ  Using local Statevector simulation for speed.")
            print(
                "         Swap Statevector for service.least_busy() to run on hardware."
            )
        except Exception as exc:
            print(f"  [!] Runtime init skipped ({exc}). Statevector only.")
    else:
        print("  [!] qiskit-ibm-runtime not installed. Statevector only.")

    # ── 1. Dataset ─────────────────────────────────────────────────────────────
    print("\n[1]  Generating Synthetic Protein Dataset")
    X, y = generate_protein_dataset(
        CFG["n_samples"], CFG["seq_len"], CFG["random_seed"]
    )
    counts = np.bincount(y)
    print(f"  Samples   : {len(X)}   Features : {X.shape[1]}")
    print(f"  Class dist: Helix={counts[0]}  Sheet={counts[1]}  Coil={counts[2]}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=CFG["test_size"],
        random_state=CFG["random_seed"],
        stratify=y,
    )
    print(f"  Train/Test: {len(X_tr)} / {len(X_te)}")

    # ── 2. Fixed quantum parameters (random seed for reproducibility) ──────────
    rng = np.random.default_rng(CFG["random_seed"])
    params_shallow = rng.uniform(0, 2 * np.pi, 2 * N_QUBITS)  # 1 layer
    params_deep = rng.uniform(0, 2 * np.pi, 4 * N_QUBITS)  # 2 layers

    # ── 3. Quantum feature extraction ──────────────────────────────────────────
    print("\n[2]  Quantum Feature Extraction")

    print("  ○  Shallow QNN circuit…", end=" ", flush=True)
    t0 = time.time()
    Xq_tr_sh = extract_quantum_features(
        X_tr, build_shallow_circuit, params_shallow, N_QUBITS, "shallow-train"
    )
    Xq_te_sh = extract_quantum_features(
        X_te, build_shallow_circuit, params_shallow, N_QUBITS, "shallow-test"
    )
    t_shallow = time.time() - t0
    print(f"done in {t_shallow:.2f}s  shape={Xq_tr_sh.shape}")

    print("  ○  Deep QNN circuit…  ", end=" ", flush=True)
    t0 = time.time()
    Xq_tr_dp = extract_quantum_features(
        X_tr, build_deep_circuit, params_deep, N_QUBITS, "deep-train"
    )
    Xq_te_dp = extract_quantum_features(
        X_te, build_deep_circuit, params_deep, N_QUBITS, "deep-test"
    )
    t_deep = time.time() - t0
    print(f"done in {t_deep:.2f}s  shape={Xq_tr_dp.shape}")

    # ── 4. Scale all feature sets ──────────────────────────────────────────────
    Xr_tr_s, Xr_te_s, _ = make_scaler_and_fit(X_tr, X_te)  # raw
    Xsh_tr_s, Xsh_te_s, _ = make_scaler_and_fit(Xq_tr_sh, Xq_te_sh)  # shallow Q
    Xdp_tr_s, Xdp_te_s, _ = make_scaler_and_fit(Xq_tr_dp, Xq_te_dp)  # deep Q

    # ── 5. Train & Evaluate all six models ─────────────────────────────────────
    print("\n[3]  Training & Evaluation\n")
    all_results = []
    conf_records = []

    def run(name, trainer_fn, X_train, X_test, quantum_t=0.0):
        t0 = time.time()
        clf = trainer_fn(X_train, y_tr)
        elapsed = time.time() - t0 + quantum_t
        metrics = evaluate(clf, X_test, y_te, name, elapsed)
        all_results.append(metrics)
        conf_records.append((name, y_te, clf.predict(X_test)))

    # ── Classical on raw features
    run("LR  │ Raw Features", train_lr, Xr_tr_s, Xr_te_s)
    run("MLP │ Raw Features", train_mlp, Xr_tr_s, Xr_te_s)

    # ── Classical on shallow quantum features
    run("LR  │ Shallow QNN Feats", train_lr, Xsh_tr_s, Xsh_te_s, t_shallow)
    run("MLP │ Shallow QNN Feats", train_mlp, Xsh_tr_s, Xsh_te_s, t_shallow)

    # ── Classical on deep quantum features
    run("LR  │ Deep QNN Feats", train_lr, Xdp_tr_s, Xdp_te_s, t_deep)
    run("MLP │ Deep QNN Feats", train_mlp, Xdp_tr_s, Xdp_te_s, t_deep)

    # ── 6. Comparison table ────────────────────────────────────────────────────
    df = (
        pd.DataFrame(all_results)
        .sort_values("Accuracy", ascending=False)
        .reset_index(drop=True)
    )
    df.index += 1  # 1-based rank

    print("\n" + "═" * 72)
    print("  FINAL COMPARISON TABLE  (sorted by Accuracy ↓)")
    print("═" * 72)
    print(df.to_string(float_format=lambda v: f"{v:.4f}"))
    print("═" * 72)

    # ── 7. Save CSV ────────────────────────────────────────────────────────────
    df.to_csv(CFG["output_csv"], index_label="Rank")
    print(f"\n  [✓] Results saved → {CFG['output_csv']}")

    # ── 8. Plots ───────────────────────────────────────────────────────────────
    print("\n[4]  Generating Figures")
    plot_df = df.copy()
    plot_df["Model"] = plot_df["Model"].str.replace(" │ ", "\n", regex=False)
    plot_results(plot_df, CFG["output_png"])
    plot_confusion_matrices(conf_records, CFG["output_png"])

    # ── 9. Summary ─────────────────────────────────────────────────────────────
    best = df.iloc[0]
    t_total = time.time() - t_total_start
    print(f"\n{'─' * 62}")
    print(f"  Best model : {best['Model']}")
    print(f"  Accuracy   : {best['Accuracy']:.4f}   F1 : {best['F1']:.4f}")
    print(f"  Total wall-clock time : {t_total:.1f}s")
    print(f"  Quantum feature time  : shallow={t_shallow:.2f}s  deep={t_deep:.2f}s")
    print(f"{'─' * 62}\n")


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    main()
