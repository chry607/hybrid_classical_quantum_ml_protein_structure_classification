"""
==============================================================================
  Quantum-Enhanced Learning of Protein Structural Features at Small Scale
  A Hybrid Quantum-Classical Proof-of-Concept Study  ── v2 (Balanced + ZZ)
==============================================================================

Key upgrades over v1:
  ① Balanced dataset  — noise injection + oversampling so all 3 classes are
                         equally represented (was 89% Helix, <1% Coil).
  ② ZZ correlator features — quantum feature vector expanded from n_qubits
                         to n_qubits + (n_qubits-1) by adding <ZiZj> pairs
                         that capture entanglement the classical model cannot
                         see in raw angles.
  ③ Robust evaluation — labels/target_names derived from data, never assumed.
  ④ Hardware-ready    — QiskitRuntimeService connected; Statevector used
                         locally for speed. One-line swap to real backend.

Architecture:
  Balanced Synthetic Dataset  (equal Helix / Sheet / Coil)
       |                              |
  Raw Features                Quantum Feature Map
  (4 raw angles)          Shallow QNN   |   Deep QNN
                          <Z> + <ZZ>    |   <Z> + <ZZ>
       |                              |
   LR / MLP                      LR / MLP on Q-features
       |                              |
       +-----------+  +---------------+
                   |  |
              CSV + 3 charts

Install:
    pip install qiskit qiskit-ibm-runtime scikit-learn pandas matplotlib numpy
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
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
    n_samples_per_class = 300,     # balanced: 300 x 3 = 900 total
    seq_len             = 4,       # features == qubits
    test_size           = 0.20,
    random_seed         = 42,
    noise_std           = 0.55,    # score-noise to break Helix dominance
    output_csv          = "quantum_protein_results.csv",
    output_png          = "quantum_protein_comparison.png",
    class_names         = ["Helix", "Sheet", "Coil"],
)
N_QUBITS = CFG["seq_len"]


# ==============================================================================
# 1.  DATASET -- BALANCED WITH NOISE + OVERSAMPLING
# ==============================================================================

def generate_protein_dataset(n_per_class: int, seq_len: int, seed: int,
                              noise_std: float):
    """
    Synthetic protein-like dataset with three balanced classes.

    Step 1 -- Raw angles in [0, pi] represent per-residue dihedral features.
    Step 2 -- Nonlinear scoring functions (sin/cos cross-terms) create curved,
              non-linearly separable boundaries:

        Helix score  = sum(sin(xi)) + sin(x0*x1)           correlated
        Sheet score  = sum(cos(xi)) + cos(x-1*x-2)         alternating
        Coil  score  = sum(sin(xi)*cos(xi)) + sin(sum(xi)) mixed

    Step 3 -- Gaussian noise is added to each score so no single class
              dominates (fixes the 89% Helix imbalance seen in v1).
    Step 4 -- The raw label distribution is equalised via oversampling
              so every class contributes exactly n_per_class samples.
    """
    rng   = np.random.default_rng(seed)
    n_raw = n_per_class * 10          # over-generate, then balance
    X_raw = rng.uniform(0, np.pi, size=(n_raw, seq_len))

    s = np.zeros((n_raw, 3))

    # Helix: strongly correlated sin pattern
    s[:, 0] = (np.sum(np.sin(X_raw), axis=1)
               + np.sin(X_raw[:, 0] * X_raw[:, 1])
               + 0.3 * np.sin(2 * X_raw[:, 0])
               + rng.normal(0, noise_std, n_raw))   # noise injection

    # Sheet: cosine / alternating pattern
    s[:, 1] = (np.sum(np.cos(X_raw), axis=1)
               + np.cos(X_raw[:, -1] * X_raw[:, -2])
               + 0.3 * np.cos(2 * X_raw[:, -1])
               + rng.normal(0, noise_std, n_raw))

    # Coil: nonlinear mixed
    s[:, 2] = (np.sum(np.sin(X_raw) * np.cos(X_raw), axis=1)
               + np.sin(np.sum(X_raw, axis=1))
               + 0.3 * np.cos(X_raw[:, 0] - X_raw[:, -1])
               + rng.normal(0, noise_std, n_raw))

    y_raw = np.argmax(s, axis=1)

    # Oversample each class to exactly n_per_class rows
    X_parts, y_parts = [], []
    for cls in range(3):
        idx = np.where(y_raw == cls)[0]
        idx_resampled = resample(
            idx, n_samples=n_per_class,
            replace=(len(idx) < n_per_class),
            random_state=seed,
        )
        X_parts.append(X_raw[idx_resampled])
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
    Shallow variational circuit (1 variational layer).

        Encoding  : RY(xi)           -- angle encoding of input features
        Var Layer : RY(theta_i) . RZ(phi_i) -- learnable rotations
        Entangle  : linear CNOT chain q0->q1->...->qn-1

    params shape: (2 x n_qubits,)  = [theta_0..theta_n-1 | phi_0..phi_n-1]
    """
    qc = QuantumCircuit(n_qubits)
    # Angle encoding
    for i in range(n_qubits):
        qc.ry(float(x[i]), i)
    # Variational layer
    for i in range(n_qubits):
        qc.ry(float(params[i]),            i)
        qc.rz(float(params[n_qubits + i]), i)
    # Linear entanglement
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    return qc


def build_deep_circuit(x: np.ndarray, n_qubits: int,
                        params: np.ndarray) -> QuantumCircuit:
    """
    Deep variational circuit (2 variational layers + circular entanglement).

        Encoding   : RY(xi)
        Layer 1    : RY(theta1_i) . RZ(phi1_i) -> linear CNOT
        Layer 2    : RY(theta2_i) . RZ(phi2_i) -> circular CNOT (ring: qn->q0)

    params shape: (4 x n_qubits,)
    """
    qc = QuantumCircuit(n_qubits)
    L  = n_qubits
    # Angle encoding
    for i in range(n_qubits):
        qc.ry(float(x[i]), i)
    # Layer 1
    for i in range(n_qubits):
        qc.ry(float(params[i]),     i)
        qc.rz(float(params[L + i]), i)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    # Layer 2
    for i in range(n_qubits):
        qc.ry(float(params[2*L + i]), i)
        qc.rz(float(params[3*L + i]), i)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    qc.cx(n_qubits - 1, 0)          # ring closure
    return qc


# ==============================================================================
# 3.  QUANTUM FEATURE EXTRACTION  (<Z> + <ZZ> correlators)
# ==============================================================================

def build_observables(n_qubits: int) -> list:
    """
    Construct a richer observable set capturing single-qubit magnetisation
    AND two-qubit entanglement correlations.

    Single-qubit <Zi>  (n_qubits operators):
        Qiskit Pauli strings are little-endian -- qubit 0 is the rightmost char.
        Qubit i  ->  "I"*(n-1-i) + "Z" + "I"*i

    Two-qubit <Zi Zi+1>  (n_qubits-1 operators):
        Nearest-neighbour pairs capture the effect of CNOT entanglement.
        These are the correlators that distinguish quantum from classical
        feature maps -- a classical model on raw angles cannot access them.

    Total features per sample: n_qubits + (n_qubits - 1)
        e.g. 4 qubits -> 4 + 3 = 7 features  (vs 4 in v1)
    """
    ops = []
    n   = n_qubits

    # Single-qubit Z expectation values
    for i in range(n):
        s = ["I"] * n
        s[n - 1 - i] = "Z"
        ops.append(SparsePauliOp("".join(s)))

    # Nearest-neighbour ZZ correlators
    for i in range(n - 1):
        s = ["I"] * n
        s[n - 1 - i] = "Z"
        s[n - 2 - i] = "Z"
        ops.append(SparsePauliOp("".join(s)))

    return ops


def extract_quantum_features(X: np.ndarray, circuit_fn, params: np.ndarray,
                              n_qubits: int) -> np.ndarray:
    """
    Map every input sample through the quantum circuit and collect
    expectation values <Ok> for all observables Ok in build_observables().

    No backprop through the quantum layer -- params are fixed random
    initialisations (quantum random feature map / kernel perspective).
    This keeps wall-clock time well under 1 minute for 900 samples.

    To switch to real hardware, replace the Statevector block with:
        from qiskit_ibm_runtime import Estimator
        backend   = service.least_busy(operational=True, simulator=False)
        estimator = Estimator(backend)
        job       = estimator.run([(qc, observable)])
        ev        = job.result()[0].data.evs
    """
    observables = build_observables(n_qubits)
    n_feats     = len(observables)
    features    = np.zeros((len(X), n_feats))

    for idx, x in enumerate(X):
        qc = circuit_fn(x, n_qubits, params)
        sv = Statevector.from_instruction(qc)
        for k, op in enumerate(observables):
            features[idx, k] = float(sv.expectation_value(op).real)

    return features


# ==============================================================================
# 4.  CLASSICAL HELPERS
# ==============================================================================

def scale(X_tr: np.ndarray, X_te: np.ndarray):
    """Fit StandardScaler on train, apply to test (no data leakage)."""
    sc = StandardScaler()
    return sc.fit_transform(X_tr), sc.transform(X_te)


def train_lr(X_tr, y_tr):
    """
    Multinomial Logistic Regression with L-BFGS.
    class_weight='balanced' guards against any residual imbalance.
    """
    clf = LogisticRegression(
        max_iter=800, solver="lbfgs",
        class_weight="balanced",
        random_state=CFG["random_seed"],
    )
    clf.fit(X_tr, y_tr)
    return clf


def train_mlp(X_tr, y_tr):
    """
    Two-hidden-layer MLP (64 -> 32, ReLU, Adam).
    early_stopping guards against over-fitting on the small dataset.
    """
    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=800,
        random_state=CFG["random_seed"],
        early_stopping=True,
        n_iter_no_change=25,
    )
    clf.fit(X_tr, y_tr)
    return clf


# ==============================================================================
# 5.  EVALUATION
# ==============================================================================

def evaluate(clf, X_te: np.ndarray, y_te: np.ndarray,
             label: str, elapsed: float) -> dict:
    """
    Print per-class classification report + summary box.
    Returns a metrics dict for the comparison table and CSV.

    Labels and target_names are derived from actual predictions, not assumed,
    so the function is robust even when a class is absent from a small split.
    """
    y_pred        = clf.predict(X_te)
    unique_labels = sorted(np.unique(np.concatenate([y_te, y_pred])))
    tgt_names     = [CFG["class_names"][i] for i in unique_labels]

    acc                  = accuracy_score(y_te, y_pred)
    prec, rec, f1, _     = precision_recall_fscore_support(
        y_te, y_pred, average="weighted", zero_division=0,
    )

    bar = "-" * 60
    print(f"\n  +{bar}+")
    print(f"  |  {label:<56s}  |")
    print(f"  +{bar}+")
    print(f"  |  Accuracy  : {acc:>7.4f}                                         |")
    print(f"  |  Precision : {prec:>7.4f}   Recall : {rec:>7.4f}                    |")
    print(f"  |  F1-Score  : {f1:>7.4f}   Time   : {elapsed:>6.2f}s                    |")
    print(f"  +{bar}+")
    print()
    print(classification_report(
        y_te, y_pred,
        labels=unique_labels,
        target_names=tgt_names,
        zero_division=0,
    ))

    return {
        "Model"     : label,
        "Accuracy"  : round(acc,  4),
        "Precision" : round(prec, 4),
        "Recall"    : round(rec,  4),
        "F1"        : round(f1,   4),
        "Time_s"    : round(elapsed, 3),
    }


# ==============================================================================
# 6.  VISUALISATION
# ==============================================================================

PALETTE = ["#1565C0", "#2E7D32", "#6A1B9A", "#E65100", "#00695C", "#AD1457"]


def plot_results(df: pd.DataFrame, save_path: str):
    """
    Three-panel comparison figure:
      (A) Accuracy  (B) Weighted F1  (C) Wall-clock time
    """
    labels  = df["Model"].tolist()
    n       = len(labels)
    colours = PALETTE[:n]
    x_pos   = np.arange(n)
    w       = 0.58

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle(
        "Quantum-Enhanced Protein Structure Learning -- Model Comparison"
        " (v2: Balanced + ZZ correlators)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    def _bar(ax, values, title, ylabel, ylim, fmt):
        bars = ax.bar(x_pos, values, width=w, color=colours,
                      alpha=0.88, edgecolor="white", linewidth=1.2)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8.5)
        ax.set_ylim(*ylim)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015 * ylim[1],
                fmt.format(val),
                ha="center", va="bottom", fontsize=8.5, fontweight="bold",
            )

    _bar(axes[0], df["Accuracy"].tolist(), "Accuracy",          "Accuracy",  (0, 1.15), "{:.3f}")
    _bar(axes[1], df["F1"].tolist(),       "Weighted F1-Score", "F1",        (0, 1.15), "{:.3f}")
    _bar(axes[2], df["Time_s"].tolist(),   "Wall-clock Time",   "Seconds",
         (0, max(df["Time_s"]) * 1.40), "{:.2f}s")

    patches = [mpatches.Patch(color=colours[i], label=labels[i]) for i in range(n)]
    fig.legend(handles=patches, loc="lower center", ncol=min(n, 6),
               bbox_to_anchor=(0.5, -0.14), fontsize=8.5, frameon=False)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  [OK] Comparison chart   --> {save_path}")


def plot_confusion_matrices(records: list, save_path: str):
    """
    One confusion matrix per model.
    records: list of (label, y_true, y_pred)
    """
    n   = len(records)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.8))
    if n == 1:
        axes = [axes]

    for ax, (label, y_te, y_pred) in zip(axes, records):
        unique = sorted(np.unique(np.concatenate([y_te, y_pred])))
        tnames = [CFG["class_names"][i] for i in unique]
        cm     = confusion_matrix(y_te, y_pred, labels=unique)
        nc     = len(unique)

        im     = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(label, fontsize=8.5, fontweight="bold")
        ax.set_xticks(range(nc)); ax.set_yticks(range(nc))
        ax.set_xticklabels(tnames, rotation=30, fontsize=8)
        ax.set_yticklabels(tnames, fontsize=8)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        thresh = cm.max() / 2
        for i in range(nc):
            for j in range(nc):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center", fontsize=11, fontweight="bold",
                        color="white" if cm[i, j] > thresh else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Confusion Matrices (v2: Balanced Dataset)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    conf_path = save_path.replace(".png", "_confusion.png")
    fig.savefig(conf_path, dpi=150, bbox_inches="tight")
    print(f"  [OK] Confusion matrices --> {conf_path}")


def plot_class_distribution(y_before: np.ndarray, y_after: np.ndarray,
                             save_path: str):
    """
    Side-by-side bar chart: raw (imbalanced) vs balanced class distribution.
    Immediately shows readers what the fix achieved.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    names = CFG["class_names"]
    cols  = ["#E53935", "#43A047", "#1E88E5"]

    for ax, y, title in [
        (ax1, y_before, "v1: Raw Distribution (imbalanced)"),
        (ax2, y_after,  "v2: Balanced Distribution"),
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
    dist_path = save_path.replace(".png", "_class_dist.png")
    fig.savefig(dist_path, dpi=150, bbox_inches="tight")
    print(f"  [OK] Class distribution --> {dist_path}")


# ==============================================================================
# 7.  MAIN PIPELINE
# ==============================================================================

def main():
    t_total_start = time.time()

    # Banner
    print("\n" + "=" * 64)
    print("  Quantum-Enhanced Protein Structure Learning  -- v2")
    print("  Balanced Dataset + ZZ Correlator Features")
    print("=" * 64)

    # ── [0] QiskitRuntimeService ───────────────────────────────────────────────
    print("\n[0]  Runtime Setup")
    if RUNTIME_AVAILABLE:
        try:
            service  = QiskitRuntimeService()
            backends = service.backends()
            print(f"  [OK] QiskitRuntimeService connected.")
            print(f"       Backends : {[b.name for b in backends[:4]]} ...")
            print("       Using local Statevector simulation for speed.")
            print("       To run on real hardware, replace Statevector with:")
            print("         from qiskit_ibm_runtime import Estimator")
            print("         backend   = service.least_busy(operational=True, simulator=False)")
            print("         estimator = Estimator(backend)")
        except Exception as exc:
            print(f"  [!] Runtime skipped ({exc}). Statevector only.")
    else:
        print("  [!] qiskit-ibm-runtime not installed. Statevector only.")

    # ── [1] Dataset ────────────────────────────────────────────────────────────
    print("\n[1]  Generating Balanced Synthetic Protein Dataset")

    # Build raw (imbalanced) version just for the distribution comparison plot
    _rng  = np.random.default_rng(CFG["random_seed"])
    _Xraw = _rng.uniform(0, np.pi, size=(CFG["n_samples_per_class"] * 10, CFG["seq_len"]))
    _s    = np.column_stack([
        np.sum(np.sin(_Xraw), axis=1) + np.sin(_Xraw[:, 0] * _Xraw[:, 1]),
        np.sum(np.cos(_Xraw), axis=1) + np.cos(_Xraw[:, -1] * _Xraw[:, -2]),
        np.sum(np.sin(_Xraw) * np.cos(_Xraw), axis=1) + np.sin(np.sum(_Xraw, axis=1)),
    ])
    y_imbalanced = np.argmax(_s, axis=1)

    # Balanced dataset
    X, y = generate_protein_dataset(
        CFG["n_samples_per_class"], CFG["seq_len"],
        CFG["random_seed"], CFG["noise_std"],
    )
    counts = np.bincount(y)
    n_qfeats = N_QUBITS + (N_QUBITS - 1)

    print(f"  Total samples  : {len(X)}   ({CFG['n_samples_per_class']} per class)")
    print(f"  Class dist     : Helix={counts[0]}  Sheet={counts[1]}  Coil={counts[2]}")
    print(f"  Raw features   : {N_QUBITS} angles")
    print(f"  Quantum feats  : {n_qfeats}  ({N_QUBITS} x <Z> + {N_QUBITS-1} x <ZZ>)")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size    = CFG["test_size"],
        random_state = CFG["random_seed"],
        stratify     = y,
    )
    print(f"  Train / Test   : {len(X_tr)} / {len(X_te)}")

    # ── [2] Fixed random quantum parameters ────────────────────────────────────
    rng            = np.random.default_rng(CFG["random_seed"])
    params_shallow = rng.uniform(0, 2 * np.pi, 2 * N_QUBITS)  # 1 layer: 8 params
    params_deep    = rng.uniform(0, 2 * np.pi, 4 * N_QUBITS)  # 2 layers: 16 params

    # ── [3] Quantum feature extraction ────────────────────────────────────────
    print("\n[2]  Quantum Feature Extraction  (<Z> + <ZZ> correlators)")

    print(f"  o  Shallow QNN...", end=" ", flush=True)
    t0 = time.time()
    Xq_tr_sh = extract_quantum_features(X_tr, build_shallow_circuit, params_shallow, N_QUBITS)
    Xq_te_sh = extract_quantum_features(X_te, build_shallow_circuit, params_shallow, N_QUBITS)
    t_shallow = time.time() - t0
    print(f"done in {t_shallow:.2f}s  shape={Xq_tr_sh.shape}  "
          f"({N_QUBITS} <Z> + {N_QUBITS-1} <ZZ>)")

    print(f"  o  Deep QNN...  ", end=" ", flush=True)
    t0 = time.time()
    Xq_tr_dp = extract_quantum_features(X_tr, build_deep_circuit, params_deep,    N_QUBITS)
    Xq_te_dp = extract_quantum_features(X_te, build_deep_circuit, params_deep,    N_QUBITS)
    t_deep = time.time() - t0
    print(f"done in {t_deep:.2f}s  shape={Xq_tr_dp.shape}  "
          f"({N_QUBITS} <Z> + {N_QUBITS-1} <ZZ>)")

    # ── [4] Scale all feature sets ─────────────────────────────────────────────
    Xr_tr,  Xr_te  = scale(X_tr,      X_te)       # raw
    Xsh_tr, Xsh_te = scale(Xq_tr_sh,  Xq_te_sh)   # shallow Q
    Xdp_tr, Xdp_te = scale(Xq_tr_dp,  Xq_te_dp)   # deep Q

    # ── [5] Train & Evaluate all six models ───────────────────────────────────
    print("\n[3]  Training & Evaluation\n")
    all_results  = []
    conf_records = []

    def run(name, trainer_fn, X_train, X_test, quantum_t=0.0):
        t0      = time.time()
        clf     = trainer_fn(X_train, y_tr)
        elapsed = time.time() - t0 + quantum_t
        metrics = evaluate(clf, X_test, y_te, name, elapsed)
        all_results.append(metrics)
        conf_records.append((name, y_te, clf.predict(X_test)))

    run("LR  | Raw Features",     train_lr,  Xr_tr,  Xr_te)
    run("MLP | Raw Features",     train_mlp, Xr_tr,  Xr_te)
    run("LR  | Shallow QNN+ZZ",  train_lr,  Xsh_tr, Xsh_te, t_shallow)
    run("MLP | Shallow QNN+ZZ",  train_mlp, Xsh_tr, Xsh_te, t_shallow)
    run("LR  | Deep QNN+ZZ",     train_lr,  Xdp_tr, Xdp_te, t_deep)
    run("MLP | Deep QNN+ZZ",     train_mlp, Xdp_tr, Xdp_te, t_deep)

    # ── [6] Final comparison table ────────────────────────────────────────────
    df = (pd.DataFrame(all_results)
            .sort_values("Accuracy", ascending=False)
            .reset_index(drop=True))
    df.index += 1

    df["Type"] = df["Model"].apply(
        lambda m: "Quantum-Enhanced" if "QNN" in m else "Classical Baseline"
    )

    print("\n" + "=" * 82)
    print("  FINAL COMPARISON TABLE  (sorted by Accuracy, descending)")
    print("=" * 82)
    print(df[["Model", "Type", "Accuracy", "Precision", "Recall", "F1", "Time_s"]]
            .to_string(float_format=lambda v: f"{v:.4f}"))
    print("=" * 82)

    # Summary delta
    best_classical = df[df["Type"] == "Classical Baseline"]["Accuracy"].max()
    best_quantum   = df[df["Type"] == "Quantum-Enhanced"]["Accuracy"].max()
    delta          = best_quantum - best_classical
    sign           = "+" if delta >= 0 else ""
    print(f"\n  Best Classical Accuracy : {best_classical:.4f}")
    print(f"  Best Quantum  Accuracy  : {best_quantum:.4f}")
    print(f"  Delta (quantum - classical) : {sign}{delta:.4f}  "
          + ("  <-- quantum advantage!" if delta > 0 else "  <-- classical still leads"))

    # ── [7] Save CSV ──────────────────────────────────────────────────────────
    df.to_csv(CFG["output_csv"], index_label="Rank")
    print(f"\n  [OK] Results saved --> {CFG['output_csv']}")

    # ── [8] Plots ─────────────────────────────────────────────────────────────
    print("\n[4]  Generating Figures")
    plot_df = df.copy()
    plot_df["Model"] = plot_df["Model"].str.replace(" | ", "\n", regex=False)
    plot_results(plot_df, CFG["output_png"])
    plot_confusion_matrices(conf_records, CFG["output_png"])
    plot_class_distribution(y_imbalanced, y, CFG["output_png"])

    # ── [9] Final summary ─────────────────────────────────────────────────────
    best    = df.iloc[0]
    t_total = time.time() - t_total_start
    print(f"\n{'=' * 64}")
    print(f"  Best model    : {best['Model']}")
    print(f"  Accuracy      : {best['Accuracy']:.4f}   F1 : {best['F1']:.4f}")
    print(f"  Total time    : {t_total:.1f}s")
    print(f"  Quantum time  : shallow={t_shallow:.2f}s   deep={t_deep:.2f}s")
    print(f"{'=' * 64}\n")


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    main()