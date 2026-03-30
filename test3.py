"""
==============================================================================
  Quantum-Enhanced Learning of Protein Structural Features at Small Scale
  A Hybrid Quantum-Classical Proof-of-Concept Study  ── v3 (Optimised Params)
==============================================================================

Root-cause fix over v2
──────────────────────
v1 & v2 used FIXED RANDOM variational parameters.  A random circuit is just
a random projection — it has no reason to produce features that separate
protein classes.  This is why classical beat quantum by ~6% in v2.

v3 fix: COBYLA optimisation of the variational angles.
  • Objective  : cross-entropy of LR trained on quantum features of a small
                 hold-out validation set (200 train / 50 val samples).
  • Optimiser  : scipy.optimize.minimize(..., method="COBYLA")
    – gradient-free, so no backprop through circuits is needed
    – each evaluation: build circuits → Statevector → features → LR → loss
    – 80 iterations × ~0.15 s/eval ≈ 12 s per circuit type  (well < 1 min)
  • After optimisation the best params are used to encode all 720 train
    and 180 test samples for the final LR / MLP classifiers.

What this means scientifically
──────────────────────────────
Fixed params  → random projection  → loses class structure
Optimised params → the variational angles are tuned so that samples from
different secondary-structure classes land in distinguishable regions of
the Bloch-sphere / Hilbert-space measurement outcomes.

Pipeline (v3)
─────────────
  Balanced dataset (300 × 3)
        │                           │
  Raw features               Quantum feature map
  (4 raw angles)      Shallow QNN       Deep QNN
                     ┌──────────┐    ┌──────────┐
                     │  random  │    │  random  │  ← same as v2 (baseline)
                     │ optimised│    │ optimised│  ← NEW in v3
                     └──────────┘    └──────────┘
                      <Z> + <ZZ>      <Z> + <ZZ>
        │                           │
   LR / MLP                  LR / MLP on Q-features
        └──────────────┬────────────┘
               CSV + 4 charts

Install:
    pip install qiskit qiskit-ibm-runtime scikit-learn scipy pandas matplotlib numpy
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
import time
import warnings
import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    log_loss,
)
from sklearn.utils import resample

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp

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
    n_samples_per_class = 300,      # 300 × 3 = 900 total
    seq_len             = 4,        # features == qubits
    test_size           = 0.20,     # 80 / 20 split
    random_seed         = 42,
    noise_std           = 0.55,     # score noise to balance classes
    # COBYLA optimisation settings
    opt_train_size      = 200,      # samples used inside the objective
    opt_val_size        = 60,       # hold-out inside objective
    opt_maxiter         = 80,       # COBYLA iterations per circuit
    opt_rhobeg          = 0.4,      # COBYLA initial step size
    output_csv          = "quantum_protein_results.csv",
    output_png          = "quantum_protein_comparison.png",
    class_names         = ["Helix", "Sheet", "Coil"],
)
N_QUBITS = CFG["seq_len"]


# ==============================================================================
# 1.  DATASET
# ==============================================================================

def generate_protein_dataset(n_per_class: int, seq_len: int,
                              seed: int, noise_std: float):
    """
    Balanced synthetic protein dataset.

    Nonlinear sin/cos scoring + Gaussian noise ensures all three classes
    (Helix, Sheet, Coil) are equally represented after oversampling.
    See v2 docstring for full derivation.
    """
    rng   = np.random.default_rng(seed)
    n_raw = n_per_class * 10
    X_raw = rng.uniform(0, np.pi, (n_raw, seq_len))

    s = np.zeros((n_raw, 3))
    s[:, 0] = (np.sum(np.sin(X_raw), axis=1)
               + np.sin(X_raw[:, 0] * X_raw[:, 1])
               + 0.3 * np.sin(2 * X_raw[:, 0])
               + rng.normal(0, noise_std, n_raw))
    s[:, 1] = (np.sum(np.cos(X_raw), axis=1)
               + np.cos(X_raw[:, -1] * X_raw[:, -2])
               + 0.3 * np.cos(2 * X_raw[:, -1])
               + rng.normal(0, noise_std, n_raw))
    s[:, 2] = (np.sum(np.sin(X_raw) * np.cos(X_raw), axis=1)
               + np.sin(np.sum(X_raw, axis=1))
               + 0.3 * np.cos(X_raw[:, 0] - X_raw[:, -1])
               + rng.normal(0, noise_std, n_raw))

    y_raw = np.argmax(s, axis=1)

    X_parts, y_parts = [], []
    for cls in range(3):
        idx = np.where(y_raw == cls)[0]
        idx_r = resample(idx, n_samples=n_per_class,
                         replace=(len(idx) < n_per_class),
                         random_state=seed)
        X_parts.append(X_raw[idx_r])
        y_parts.append(np.full(n_per_class, cls))

    X    = np.vstack(X_parts)
    y    = np.concatenate(y_parts)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


# ==============================================================================
# 2.  QUANTUM CIRCUITS
# ==============================================================================

def build_shallow_circuit(x: np.ndarray, n_qubits: int,
                           params: np.ndarray) -> QuantumCircuit:
    """
    1-layer variational circuit.
      Encode  : RY(x_i)
      Variate : RY(theta_i) · RZ(phi_i)
      Entangle: linear CNOT chain
    params: shape (2 × n_qubits,)
    """
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.ry(float(x[i]), i)
    for i in range(n_qubits):
        qc.ry(float(params[i]),            i)
        qc.rz(float(params[n_qubits + i]), i)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    return qc


def build_deep_circuit(x: np.ndarray, n_qubits: int,
                        params: np.ndarray) -> QuantumCircuit:
    """
    2-layer variational circuit.
      Encode    : RY(x_i)
      Layer 1   : RY · RZ → linear CNOT
      Layer 2   : RY · RZ → circular CNOT (ring closure)
    params: shape (4 × n_qubits,)
    """
    qc = QuantumCircuit(n_qubits)
    L  = n_qubits
    for i in range(n_qubits):
        qc.ry(float(x[i]), i)
    for i in range(n_qubits):
        qc.ry(float(params[i]),     i)
        qc.rz(float(params[L + i]), i)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    for i in range(n_qubits):
        qc.ry(float(params[2*L + i]), i)
        qc.rz(float(params[3*L + i]), i)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    qc.cx(n_qubits - 1, 0)
    return qc


# ==============================================================================
# 3.  OBSERVABLES  (<Z> + nearest-neighbour <ZZ>)
# ==============================================================================

def build_observables(n_qubits: int) -> list:
    """
    n_qubits single-qubit <Z_i> operators  +
    (n_qubits-1) two-qubit <Z_i Z_{i+1}> correlators.

    Total: 2*n_qubits - 1  features per sample.
    The ZZ correlators are the signature of entanglement — they encode
    information that a classical model on raw angles cannot reconstruct.
    """
    ops, n = [], n_qubits
    for i in range(n):                      # <Z_i>
        s = ["I"] * n; s[n - 1 - i] = "Z"
        ops.append(SparsePauliOp("".join(s)))
    for i in range(n - 1):                  # <Z_i Z_{i+1}>
        s = ["I"] * n
        s[n - 1 - i] = "Z"; s[n - 2 - i] = "Z"
        ops.append(SparsePauliOp("".join(s)))
    return ops


def extract_quantum_features(X: np.ndarray, circuit_fn,
                              params: np.ndarray, n_qubits: int) -> np.ndarray:
    """
    Run every sample through the circuit, collect <O_k> expectation values.
    Uses local Statevector simulation (fast, exact).
    Hardware swap: replace sv.expectation_value(op) with Estimator job.
    """
    observables = build_observables(n_qubits)
    features    = np.zeros((len(X), len(observables)))
    for idx, x in enumerate(X):
        sv = Statevector.from_instruction(circuit_fn(x, n_qubits, params))
        for k, op in enumerate(observables):
            features[idx, k] = float(sv.expectation_value(op).real)
    return features


# ==============================================================================
# 4.  PARAMETER OPTIMISATION  (COBYLA — gradient-free)
# ==============================================================================

def _cross_entropy_objective(params: np.ndarray, circuit_fn,
                              X_opt: np.ndarray, y_opt: np.ndarray,
                              X_val: np.ndarray, y_val: np.ndarray,
                              n_qubits: int) -> float:
    """
    Objective function for COBYLA.

    1. Extract quantum features for the optimisation train set using `params`
    2. Fit a fast LR on those features
    3. Return cross-entropy loss on the validation set

    Cross-entropy is smooth and gives COBYLA a meaningful gradient signal
    even when accuracy is flat (avoids the plateau problem).
    """
    sc        = StandardScaler()
    Xq_opt    = extract_quantum_features(X_opt, circuit_fn, params, n_qubits)
    Xq_val    = extract_quantum_features(X_val, circuit_fn, params, n_qubits)
    Xq_opt_s  = sc.fit_transform(Xq_opt)
    Xq_val_s  = sc.transform(Xq_val)

    clf = LogisticRegression(max_iter=300, solver="lbfgs",
                             class_weight="balanced", random_state=0)
    clf.fit(Xq_opt_s, y_opt)
    proba = clf.predict_proba(Xq_val_s)
    return log_loss(y_val, proba)


def optimise_params(circuit_fn, init_params: np.ndarray,
                    X_opt: np.ndarray, y_opt: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray,
                    n_qubits: int, label: str) -> np.ndarray:
    """
    Run COBYLA to minimise cross-entropy over variational parameters.

    Why COBYLA?
      • Gradient-free — no parameter-shift rule needed
      • Handles box-constrained angles naturally
      • Converges in O(n_params²) evaluations — fast for 8–16 params
      • scipy.optimize.minimize wraps it natively

    Returns the optimised parameter vector.
    """
    print(f"    Optimising {label} ({len(init_params)} params, "
          f"max {CFG['opt_maxiter']} iters)...")

    iteration  = [0]
    best_loss  = [np.inf]
    start      = time.time()

    def callback_obj(p):
        loss = _cross_entropy_objective(
            p, circuit_fn, X_opt, y_opt, X_val, y_val, n_qubits)
        iteration[0] += 1
        if loss < best_loss[0]:
            best_loss[0] = loss
        # Progress every 10 iters
        if iteration[0] % 10 == 0:
            elapsed = time.time() - start
            print(f"      iter {iteration[0]:3d}  loss={loss:.4f}  "
                  f"best={best_loss[0]:.4f}  t={elapsed:.1f}s")
        return loss

    result = opt.minimize(
        callback_obj,
        init_params,
        method  = "COBYLA",
        options = {
            "maxiter" : CFG["opt_maxiter"],
            "rhobeg"  : CFG["opt_rhobeg"],
        },
    )
    elapsed = time.time() - start
    print(f"    Done. Final loss={result.fun:.4f}  time={elapsed:.1f}s")
    return result.x


# ==============================================================================
# 5.  CLASSICAL HELPERS
# ==============================================================================

def scale(X_tr: np.ndarray, X_te: np.ndarray):
    sc = StandardScaler()
    return sc.fit_transform(X_tr), sc.transform(X_te)


def train_lr(X_tr, y_tr):
    clf = LogisticRegression(max_iter=800, solver="lbfgs",
                             class_weight="balanced",
                             random_state=CFG["random_seed"])
    clf.fit(X_tr, y_tr)
    return clf


def train_mlp(X_tr, y_tr):
    clf = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                        solver="adam", max_iter=800,
                        random_state=CFG["random_seed"],
                        early_stopping=True, n_iter_no_change=25)
    clf.fit(X_tr, y_tr)
    return clf


# ==============================================================================
# 6.  EVALUATION
# ==============================================================================

def evaluate(clf, X_te: np.ndarray, y_te: np.ndarray,
             label: str, elapsed: float) -> dict:
    y_pred        = clf.predict(X_te)
    unique_labels = sorted(np.unique(np.concatenate([y_te, y_pred])))
    tgt_names     = [CFG["class_names"][i] for i in unique_labels]
    acc           = accuracy_score(y_te, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_te, y_pred, average="weighted", zero_division=0)

    bar = "-" * 62
    print(f"\n  +{bar}+")
    print(f"  |  {label:<58s}  |")
    print(f"  +{bar}+")
    print(f"  |  Accuracy  : {acc:>7.4f}                                             |")
    print(f"  |  Precision : {prec:>7.4f}   Recall : {rec:>7.4f}                        |")
    print(f"  |  F1-Score  : {f1:>7.4f}   Time   : {elapsed:>6.2f}s                        |")
    print(f"  +{bar}+")
    print()
    print(classification_report(y_te, y_pred, labels=unique_labels,
                                target_names=tgt_names, zero_division=0))
    return {"Model": label, "Accuracy": round(acc, 4),
            "Precision": round(prec, 4), "Recall": round(rec, 4),
            "F1": round(f1, 4), "Time_s": round(elapsed, 3)}


# ==============================================================================
# 7.  VISUALISATION
# ==============================================================================

PALETTE = ["#1565C0", "#2E7D32", "#B71C1C", "#E65100",
           "#6A1B9A", "#00695C", "#AD1457", "#F57F17",
           "#00838F", "#5E35B1", "#C62828", "#FF6F00"]


def _bar_panel(ax, x_pos, values, labels, colours, title, ylabel, ylim, fmt):
    bars = ax.bar(x_pos, values, width=0.55, color=colours,
                  alpha=0.88, edgecolor="white", linewidth=1.2)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(*ylim)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012 * ylim[1],
                fmt.format(val), ha="center", va="bottom",
                fontsize=8, fontweight="bold")


def plot_results(df: pd.DataFrame, save_path: str):
    labels  = df["Model"].tolist()
    n       = len(labels)
    colours = PALETTE[:n]
    x_pos   = np.arange(n)

    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    fig.suptitle(
        "Quantum-Enhanced Protein Structure Learning — v3: Optimised Variational Params",
        fontsize=13, fontweight="bold", y=1.01)

    _bar_panel(axes[0], x_pos, df["Accuracy"].tolist(), labels, colours,
               "Accuracy", "Accuracy", (0, 1.15), "{:.3f}")
    _bar_panel(axes[1], x_pos, df["F1"].tolist(), labels, colours,
               "Weighted F1", "F1", (0, 1.15), "{:.3f}")
    _bar_panel(axes[2], x_pos, df["Time_s"].tolist(), labels, colours,
               "Wall-clock Time (incl. optimisation)", "Seconds",
               (0, max(df["Time_s"]) * 1.4), "{:.1f}s")

    patches = [mpatches.Patch(color=colours[i], label=labels[i]) for i in range(n)]
    fig.legend(handles=patches, loc="lower center", ncol=min(n, 4),
               bbox_to_anchor=(0.5, -0.18), fontsize=8, frameon=False)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  [OK] Comparison chart      --> {save_path}")


def plot_confusion_matrices(records: list, save_path: str):
    n   = len(records)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(5 * ((n + 1) // 2), 9.5))
    axes = axes.flatten()
    for ax in axes[n:]:
        ax.set_visible(False)

    for ax, (label, y_te, y_pred) in zip(axes, records):
        unique = sorted(np.unique(np.concatenate([y_te, y_pred])))
        tnames = [CFG["class_names"][i] for i in unique]
        cm     = confusion_matrix(y_te, y_pred, labels=unique)
        nc     = len(unique)
        im     = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(label, fontsize=8, fontweight="bold")
        ax.set_xticks(range(nc)); ax.set_yticks(range(nc))
        ax.set_xticklabels(tnames, rotation=30, fontsize=7.5)
        ax.set_yticklabels(tnames, fontsize=7.5)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        thresh = cm.max() / 2
        for i in range(nc):
            for j in range(nc):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="white" if cm[i, j] > thresh else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Confusion Matrices — v3 (Random vs Optimised Params)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    p = save_path.replace(".png", "_confusion.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"  [OK] Confusion matrices    --> {p}")


def plot_optimisation_curves(curves: dict, save_path: str):
    """
    Loss curves recorded during COBYLA optimisation for each circuit type.
    Shows that the optimiser is actually improving the quantum feature quality.
    """
    fig, axes = plt.subplots(1, len(curves), figsize=(7 * len(curves), 4))
    if len(curves) == 1:
        axes = [axes]
    colours_c = {"Shallow QNN": "#1565C0", "Deep QNN": "#B71C1C"}

    for ax, (name, losses) in zip(axes, curves.items()):
        col = colours_c.get(name, "#333333")
        ax.plot(range(1, len(losses) + 1), losses, color=col,
                linewidth=2, alpha=0.9)
        ax.fill_between(range(1, len(losses) + 1), losses,
                        alpha=0.15, color=col)
        # Mark best
        best_iter = int(np.argmin(losses)) + 1
        best_val  = min(losses)
        ax.axvline(best_iter, color=col, linestyle="--", alpha=0.6)
        ax.annotate(f"best={best_val:.3f}\n@ iter {best_iter}",
                    xy=(best_iter, best_val),
                    xytext=(best_iter + max(1, len(losses) // 8), best_val + 0.02),
                    fontsize=9, color=col,
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.2))
        ax.set_title(f"COBYLA Loss Curve — {name}", fontweight="bold", fontsize=11)
        ax.set_xlabel("Objective evaluation")
        ax.set_ylabel("Cross-entropy loss (validation)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(linestyle="--", alpha=0.3)

    fig.suptitle("Quantum Parameter Optimisation Progress (v3)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    p = save_path.replace(".png", "_opt_curves.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"  [OK] Optimisation curves   --> {p}")


def plot_class_distribution(y_before: np.ndarray, y_after: np.ndarray,
                             save_path: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    names = CFG["class_names"]
    cols  = ["#E53935", "#43A047", "#1E88E5"]
    for ax, y, title in [
        (ax1, y_before, "v1: Raw Distribution (imbalanced)"),
        (ax2, y_after,  "v3: Balanced Distribution"),
    ]:
        counts = [int(np.sum(y == i)) for i in range(3)]
        bars   = ax.bar(names, counts, color=cols, alpha=0.85, edgecolor="white")
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_ylabel("Sample count")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 5, str(cnt),
                    ha="center", fontsize=11, fontweight="bold")
    plt.tight_layout()
    p = save_path.replace(".png", "_class_dist.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"  [OK] Class distribution    --> {p}")


# ==============================================================================
# 8.  MAIN PIPELINE
# ==============================================================================

def main():
    t_total_start = time.time()

    print("\n" + "=" * 66)
    print("  Quantum-Enhanced Protein Structure Learning  -- v3")
    print("  Optimised Variational Parameters via COBYLA")
    print("=" * 66)

    # ── [0] Runtime ────────────────────────────────────────────────────────────
    print("\n[0]  Runtime Setup")
    if RUNTIME_AVAILABLE:
        try:
            service  = QiskitRuntimeService()
            backends = service.backends()
            print(f"  [OK] QiskitRuntimeService connected.")
            print(f"       Backends : {[b.name for b in backends[:4]]} ...")
            print("       Statevector simulation used locally (fast, exact).")
            print("       Hardware swap: see extract_quantum_features() docstring.")
        except Exception as exc:
            print(f"  [!] Runtime skipped ({exc}). Statevector only.")
    else:
        print("  [!] qiskit-ibm-runtime not installed. Statevector only.")

    # ── [1] Dataset ────────────────────────────────────────────────────────────
    print("\n[1]  Generating Balanced Dataset")

    # Raw imbalanced — for the dist plot only
    _rng  = np.random.default_rng(CFG["random_seed"])
    _Xraw = _rng.uniform(0, np.pi, (CFG["n_samples_per_class"] * 10, CFG["seq_len"]))
    _s    = np.column_stack([
        np.sum(np.sin(_Xraw), axis=1) + np.sin(_Xraw[:, 0] * _Xraw[:, 1]),
        np.sum(np.cos(_Xraw), axis=1) + np.cos(_Xraw[:, -1] * _Xraw[:, -2]),
        np.sum(np.sin(_Xraw) * np.cos(_Xraw), axis=1) + np.sin(np.sum(_Xraw, axis=1)),
    ])
    y_imbalanced = np.argmax(_s, axis=1)

    X, y = generate_protein_dataset(
        CFG["n_samples_per_class"], CFG["seq_len"],
        CFG["random_seed"], CFG["noise_std"])
    counts   = np.bincount(y)
    n_qfeats = N_QUBITS + (N_QUBITS - 1)

    print(f"  Total         : {len(X)}   ({CFG['n_samples_per_class']} per class)")
    print(f"  Class dist    : Helix={counts[0]}  Sheet={counts[1]}  Coil={counts[2]}")
    print(f"  Raw features  : {N_QUBITS} angles  |  "
          f"Quantum features: {n_qfeats}  ({N_QUBITS}x<Z> + {N_QUBITS-1}x<ZZ>)")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=CFG["test_size"],
        random_state=CFG["random_seed"], stratify=y)
    print(f"  Train / Test  : {len(X_tr)} / {len(X_te)}")

    # ── [2] Optimisation subset split ──────────────────────────────────────────
    # Carve out a small stratified subset from the TRAIN split for COBYLA.
    # The test set is never touched during optimisation.
    opt_size = CFG["opt_train_size"] + CFG["opt_val_size"]
    X_opt_all, _, y_opt_all, _ = train_test_split(
        X_tr, y_tr,
        train_size   = opt_size,
        random_state = CFG["random_seed"],
        stratify     = y_tr,
    )
    X_opt, X_val_opt, y_opt, y_val_opt = train_test_split(
        X_opt_all, y_opt_all,
        train_size   = CFG["opt_train_size"],
        random_state = CFG["random_seed"],
        stratify     = y_opt_all,
    )
    print(f"\n  Optimisation split: {len(X_opt)} train / {len(X_val_opt)} val "
          f"(inside COBYLA objective, test set untouched)")

    # ── [3] Random seed params  (same as v2 — kept as baseline) ───────────────
    rng            = np.random.default_rng(CFG["random_seed"])
    params_sh_rand = rng.uniform(0, 2 * np.pi, 2 * N_QUBITS)
    params_dp_rand = rng.uniform(0, 2 * np.pi, 4 * N_QUBITS)

    # ── [4] Optimise params with COBYLA ────────────────────────────────────────
    print("\n[2]  COBYLA Optimisation of Variational Parameters")
    print("  (minimises cross-entropy of LR trained on quantum features)")

    opt_curves: dict = {}

    # Shallow — wrap objective to record loss curve
    sh_losses: list = []
    def sh_objective(p):
        l = _cross_entropy_objective(
            p, build_shallow_circuit, X_opt, y_opt, X_val_opt, y_val_opt, N_QUBITS)
        sh_losses.append(l)
        return l

    t0 = time.time()
    print("\n  Shallow QNN:")
    res_sh = opt.minimize(sh_objective, params_sh_rand.copy(),
                          method="COBYLA",
                          options={"maxiter": CFG["opt_maxiter"],
                                   "rhobeg":  CFG["opt_rhobeg"]})
    t_opt_sh = time.time() - t0
    params_sh_opt = res_sh.x
    opt_curves["Shallow QNN"] = sh_losses
    print(f"  Shallow done: final_loss={res_sh.fun:.4f}  "
          f"evals={len(sh_losses)}  time={t_opt_sh:.1f}s")

    # Deep — same wrapper
    dp_losses: list = []
    def dp_objective(p):
        l = _cross_entropy_objective(
            p, build_deep_circuit, X_opt, y_opt, X_val_opt, y_val_opt, N_QUBITS)
        dp_losses.append(l)
        return l

    t0 = time.time()
    print("\n  Deep QNN:")
    res_dp = opt.minimize(dp_objective, params_dp_rand.copy(),
                          method="COBYLA",
                          options={"maxiter": CFG["opt_maxiter"],
                                   "rhobeg":  CFG["opt_rhobeg"]})
    t_opt_dp = time.time() - t0
    params_dp_opt = res_dp.x
    opt_curves["Deep QNN"] = dp_losses
    print(f"  Deep done:    final_loss={res_dp.fun:.4f}  "
          f"evals={len(dp_losses)}  time={t_opt_dp:.1f}s")

    # ── [5] Extract features with ALL four param sets ──────────────────────────
    print("\n[3]  Feature Extraction (random params  +  optimised params)")

    def extract_both(circuit_fn, p_rand, p_opt, tag_rand, tag_opt):
        print(f"  o  {tag_rand}...", end=" ", flush=True)
        t0 = time.time()
        tr_r = extract_quantum_features(X_tr, circuit_fn, p_rand, N_QUBITS)
        te_r = extract_quantum_features(X_te, circuit_fn, p_rand, N_QUBITS)
        t_r  = time.time() - t0
        print(f"{t_r:.1f}s", end="   |   ", flush=True)

        print(f"{tag_opt}...", end=" ", flush=True)
        t0 = time.time()
        tr_o = extract_quantum_features(X_tr, circuit_fn, p_opt, N_QUBITS)
        te_o = extract_quantum_features(X_te, circuit_fn, p_opt, N_QUBITS)
        t_o  = time.time() - t0
        print(f"{t_o:.1f}s")
        return (tr_r, te_r, t_r), (tr_o, te_o, t_o)

    (sh_tr_r, sh_te_r, t_sh_r), (sh_tr_o, sh_te_o, t_sh_o) = extract_both(
        build_shallow_circuit, params_sh_rand, params_sh_opt,
        "Shallow [random]", "Shallow [optimised]")

    (dp_tr_r, dp_te_r, t_dp_r), (dp_tr_o, dp_te_o, t_dp_o) = extract_both(
        build_deep_circuit, params_dp_rand, params_dp_opt,
        "Deep    [random]", "Deep    [optimised]")

    # ── [6] Scale ──────────────────────────────────────────────────────────────
    Xr_tr,    Xr_te    = scale(X_tr,    X_te)
    Xsh_r_tr, Xsh_r_te = scale(sh_tr_r, sh_te_r)
    Xsh_o_tr, Xsh_o_te = scale(sh_tr_o, sh_te_o)
    Xdp_r_tr, Xdp_r_te = scale(dp_tr_r, dp_te_r)
    Xdp_o_tr, Xdp_o_te = scale(dp_tr_o, dp_te_o)

    # ── [7] Train & Evaluate all 10 models ─────────────────────────────────────
    print("\n[4]  Training & Evaluation\n")
    all_results  = []
    conf_records = []

    def run(name, trainer_fn, X_train, X_test, extra_t=0.0):
        t0      = time.time()
        clf     = trainer_fn(X_train, y_tr)
        elapsed = time.time() - t0 + extra_t
        metrics = evaluate(clf, X_test, y_te, name, elapsed)
        all_results.append(metrics)
        conf_records.append((name, y_te, clf.predict(X_test)))

    # Classical baselines
    run("LR  | Raw",                   train_lr,  Xr_tr,    Xr_te)
    run("MLP | Raw",                   train_mlp, Xr_tr,    Xr_te)
    # Random-params quantum (v2 equivalent)
    run("LR  | Shallow [random]",      train_lr,  Xsh_r_tr, Xsh_r_te, t_sh_r)
    run("MLP | Shallow [random]",      train_mlp, Xsh_r_tr, Xsh_r_te, t_sh_r)
    run("LR  | Deep    [random]",      train_lr,  Xdp_r_tr, Xdp_r_te, t_dp_r)
    run("MLP | Deep    [random]",      train_mlp, Xdp_r_tr, Xdp_r_te, t_dp_r)
    # Optimised-params quantum (v3 new)
    run("LR  | Shallow [optimised]",   train_lr,  Xsh_o_tr, Xsh_o_te,
        t_sh_o + t_opt_sh)
    run("MLP | Shallow [optimised]",   train_mlp, Xsh_o_tr, Xsh_o_te,
        t_sh_o + t_opt_sh)
    run("LR  | Deep    [optimised]",   train_lr,  Xdp_o_tr, Xdp_o_te,
        t_dp_o + t_opt_dp)
    run("MLP | Deep    [optimised]",   train_mlp, Xdp_o_tr, Xdp_o_te,
        t_dp_o + t_opt_dp)

    # ── [8] Comparison table ───────────────────────────────────────────────────
    df = (pd.DataFrame(all_results)
            .sort_values("Accuracy", ascending=False)
            .reset_index(drop=True))
    df.index += 1

    def _label_type(m):
        if "optimised" in m: return "Quantum (Optimised)"
        if "random"    in m: return "Quantum (Random)"
        return "Classical"

    df["Type"] = df["Model"].apply(_label_type)

    print("\n" + "=" * 88)
    print("  FINAL COMPARISON TABLE  (sorted by Accuracy, descending)")
    print("=" * 88)
    print(df[["Model", "Type", "Accuracy", "Precision", "Recall", "F1", "Time_s"]]
            .to_string(float_format=lambda v: f"{v:.4f}"))
    print("=" * 88)

    # Delta summary
    acc_classical = df[df["Type"] == "Classical"]["Accuracy"].max()
    acc_q_rand    = df[df["Type"] == "Quantum (Random)"]["Accuracy"].max()
    acc_q_opt     = df[df["Type"] == "Quantum (Optimised)"]["Accuracy"].max()

    def _delta(a, b):
        d = a - b; s = "+" if d >= 0 else ""; tag = "GAIN" if d > 0 else "loss"
        return f"{s}{d:.4f}  ({tag})"

    print(f"\n  Classical Baseline       best accuracy : {acc_classical:.4f}")
    print(f"  Quantum [random params]  best accuracy : {acc_q_rand:.4f}  "
          f"  Δ vs classical = {_delta(acc_q_rand, acc_classical)}")
    print(f"  Quantum [optimised]      best accuracy : {acc_q_opt:.4f}  "
          f"  Δ vs classical = {_delta(acc_q_opt,  acc_classical)}")
    print(f"  Optimisation gain (opt vs random)      : "
          f"{_delta(acc_q_opt, acc_q_rand)}")

    # ── [9] Save CSV ───────────────────────────────────────────────────────────
    df.to_csv(CFG["output_csv"], index_label="Rank")
    print(f"\n  [OK] Results saved --> {CFG['output_csv']}")

    # ── [10] Figures ───────────────────────────────────────────────────────────
    print("\n[5]  Generating Figures")
    plot_df = df.copy()
    plot_df["Model"] = plot_df["Model"].str.replace(" | ", "\n", regex=False)
    plot_results(plot_df, CFG["output_png"])
    plot_confusion_matrices(conf_records, CFG["output_png"])
    plot_optimisation_curves(opt_curves, CFG["output_png"])
    plot_class_distribution(y_imbalanced, y, CFG["output_png"])

    # ── [11] Final summary ─────────────────────────────────────────────────────
    best    = df.iloc[0]
    t_total = time.time() - t_total_start
    print(f"\n{'=' * 66}")
    print(f"  Best model    : {best['Model']}")
    print(f"  Accuracy      : {best['Accuracy']:.4f}   F1 : {best['F1']:.4f}")
    print(f"  Total time    : {t_total:.1f}s")
    print(f"  Opt time      : Shallow={t_opt_sh:.1f}s   Deep={t_opt_dp:.1f}s")
    print(f"{'=' * 66}\n")


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    main()