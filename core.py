# =============================================================================
# core.py
# All classifier logic lives here.
#
# Exported:
#   ALL_RESULTS  – a shared list that every embedding module appends to.
#                  Each entry is a dict with keys:
#                    embedding, classifier, accuracy, report, cm
#   train_and_evaluate(name, X_train, X_test, y_train, y_test)
#                – trains SVM, NB, LR; appends results; prints a summary.
# =============================================================================

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix)
from config import CFG


# ---------------------------------------------------------------------------
# Shared results list — every embedding module calls train_and_evaluate()
# which appends to this list.  main.py reads it to build the comparison table.
# ---------------------------------------------------------------------------
ALL_RESULTS = []


# ---------------------------------------------------------------------------
# Build classifiers
# ---------------------------------------------------------------------------

def _get_classifiers():
    """Return a dict of {name: sklearn_estimator}."""
    return {
        "svm": LinearSVC(max_iter=2000, random_state=CFG.RANDOM_STATE),
        "nb":  ComplementNB(),          # works with sparse TF-IDF; for dense we adapt below
        "lr":  LogisticRegression(
                   max_iter=1000
               ),
    }


# ---------------------------------------------------------------------------
# Main training + evaluation function
# ---------------------------------------------------------------------------

def train_and_evaluate(
    embedding_name: str,
    X_train,
    X_test,
    y_train: list,
    y_test: list,
):
    """
    Trains all three classifiers on the given feature matrices and labels.
    Appends one result dict per classifier to ALL_RESULTS.

    Parameters
    ----------
    embedding_name : str   e.g. "TF-IDF", "Word2Vec", "BERT"
    X_train        : array-like  shape (n_train, n_features)
    X_test         : array-like  shape (n_test,  n_features)
    y_train        : list of label strings
    y_test         : list of label strings
    """

    classifiers = _get_classifiers()

    for clf_name, clf in classifiers.items():

        # Naïve Bayes needs non-negative values.
        # Dense embeddings can have negatives, so we shift them for NB only.
        if clf_name == "nb":
            X_tr = _make_non_negative(X_train)
            X_te = _make_non_negative(X_test)
        else:
            X_tr = X_train
            X_te = X_test

        # --- Train ---
        clf.fit(X_tr, y_train)

        # --- Predict ---
        y_pred = clf.predict(X_te)

        # --- Metrics ---
        acc  = accuracy_score(y_test, y_pred)
        report = classification_report(
            y_test, y_pred,
            labels=CFG.LABELS,
            target_names=CFG.LABELS,
            zero_division=0,
        )
        cm = confusion_matrix(y_test, y_pred)

        # --- Store ---
        ALL_RESULTS.append({
            "embedding":  embedding_name,
            "classifier": clf_name.upper(),
            "accuracy":   round(acc * 100, 2),   # stored as percentage
            "report":     report,
            "cm":         cm,
            "y_test":     y_test,
            "y_pred":     y_pred,
            "clf_object": clf,   # kept so we can refit on hard samples later
        })

        print(f"  [{embedding_name}] {clf_name.upper():3s}  →  Accuracy: {acc*100:.2f}%")


# ---------------------------------------------------------------------------
# Helper: shift dense matrix so all values are ≥ 0  (needed for NB)
# ---------------------------------------------------------------------------

import numpy as np
from scipy import sparse

def _make_non_negative(X):
    if sparse.issparse(X):
        return X   # already fine, don't touch
    else:
        X = np.array(X, dtype=np.float64)
        X[X < 0] = 0
        return X
