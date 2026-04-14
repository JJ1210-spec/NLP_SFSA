# =============================================================================
# reporting.py
# Everything related to displaying and saving results:
#   • print_comparison_table  — accuracy table across all embedding+classifier combos
#   • plot_confusion_matrices — one CM per combo, saved as PNG
#   • get_hard_samples        — extract misclassified examples
#   • retrain_on_hard_samples — refit best model on original + hard samples
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from config import CFG


# ---------------------------------------------------------------------------
# 1. Comparison table
# ---------------------------------------------------------------------------

def print_comparison_table(all_results: list):
    """
    Prints a neat ASCII table and returns a DataFrame.

    Columns: Embedding | Classifier | Accuracy(%)
    Sorted by Accuracy descending so the best combo is at the top.
    """

    rows = [
        {
            "Embedding":   r["embedding"],
            "Classifier":  r["classifier"],
            "Accuracy (%)": r["accuracy"],
        }
        for r in all_results
    ]

    df = pd.DataFrame(rows).sort_values("Accuracy (%)", ascending=False)
    df = df.reset_index(drop=True)

    print("\n" + "=" * 55)
    print("MODEL COMPARISON TABLE  (sorted by accuracy)")
    print("=" * 55)
    print(df.to_string(index=False))
    print("=" * 55)

    # Save to CSV as well
    os.makedirs(CFG.RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(CFG.RESULTS_DIR, "comparison_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved comparison table → {csv_path}")

    return df


# ---------------------------------------------------------------------------
# 2. Confusion matrices
# ---------------------------------------------------------------------------

def plot_confusion_matrices(all_results: list):
    """
    Saves a confusion-matrix PNG for every (embedding, classifier) pair.
    """

    os.makedirs(CFG.FIGURES_DIR, exist_ok=True)

    for r in all_results:
        cm   = r["cm"]
        emb  = r["embedding"]
        clf  = r["classifier"]
        acc  = r["accuracy"]

        fig, ax = plt.subplots(figsize=(5, 4))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=CFG.LABELS,
            yticklabels=CFG.LABELS,
            ax=ax,
        )

        ax.set_title(f"{emb} + {clf}  |  Acc: {acc}%", fontsize=12)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        plt.tight_layout()

        # File name: e.g.  cm_BERT_SVM.png
        fname = f"cm_{emb.replace(' ', '_')}_{clf}.png"
        fpath = os.path.join(CFG.FIGURES_DIR, fname)
        plt.savefig(fpath, dpi=150)
        plt.close()

    print(f"\nConfusion matrices saved → {CFG.FIGURES_DIR}/")


# ---------------------------------------------------------------------------
# 3. Extract hard samples  (misclassified by the best model)
# ---------------------------------------------------------------------------

def get_hard_samples(
    best_result: dict,
    test_texts: list,
) -> tuple:
    """
    Compares y_test vs y_pred from best_result and returns the texts
    that were WRONGLY classified — these are the "hard samples".

    Returns
    -------
    hard_texts  : list[str]   — the misclassified feedback strings
    hard_labels : list[str]   — their TRUE labels
    """

    y_test = best_result["y_test"]
    y_pred = best_result["y_pred"]

    hard_texts  = []
    hard_labels = []

    for text, true, pred in zip(test_texts, y_test, y_pred):
        if true != pred:
            hard_texts.append(text)
            hard_labels.append(true)

    print(f"\n[Hard Samples] Found {len(hard_texts)} misclassified examples "
          f"out of {len(y_test)} test samples "
          f"({len(hard_texts)/len(y_test)*100:.1f}%).")

    return hard_texts, hard_labels


# ---------------------------------------------------------------------------
# 4. Retrain on hard samples (simple approach: oversample them into train set)
# ---------------------------------------------------------------------------

def retrain_on_hard_samples(
    best_result: dict,
    X_train_hard: np.ndarray,
    y_train_hard: list,
    X_train_orig: np.ndarray,
    y_train_orig: list,
    X_test:       np.ndarray,
    y_test:       list,
):
    """
    Appends the hard (misclassified) samples back into the training set
    and retrains the best classifier.

    This is a simple but effective way to improve on weak spots.

    Parameters
    ----------
    best_result    : one entry from ALL_RESULTS (has the clf_object)
    X_train_hard   : embeddings of hard samples
    y_train_hard   : true labels of hard samples
    X_train_orig   : original training embeddings
    y_train_orig   : original training labels
    X_test         : test embeddings
    y_test         : test labels
    """

    from sklearn.metrics import accuracy_score, classification_report
    import numpy as np

    clf = best_result["clf_object"]

    # Stack: original train  +  hard samples (repeated once)
    X_combined = np.vstack([X_train_orig, X_train_hard])
    y_combined = list(y_train_orig) + list(y_train_hard)

    # Refit
    clf.fit(X_combined, y_combined)
    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 55)
    print("AFTER HARD-SAMPLE RETRAINING")
    print(f"Embedding:  {best_result['embedding']}")
    print(f"Classifier: {best_result['classifier']}")
    print(f"New Accuracy: {acc*100:.2f}%  "
          f"(was {best_result['accuracy']}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("=" * 55)

    return clf, y_pred
