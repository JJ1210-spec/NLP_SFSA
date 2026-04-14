# =============================================================================
# embeddings/tfidf.py
# TF-IDF vectorisation.
# This is our baseline — no external model needed.
# =============================================================================

from sklearn.feature_extraction.text import TfidfVectorizer
from core import train_and_evaluate


def run_tfidf(
    train_strings: list,   # plain joined strings from get_clean_strings()
    test_strings: list,
    y_train: list,
    y_test: list,
):
    """
    Fits a TF-IDF vectoriser on train_strings, transforms both splits,
    then runs all classifiers via core.train_and_evaluate().
    """

    print("\n[TF-IDF] Vectorising ...")

    # TF-IDF with unigrams + bigrams, 10 000 top features
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),   # include both single words and two-word combos
        max_features=10_000,
        sublinear_tf=True,    # apply log(1 + tf) scaling — helps with long texts
    )

    # Fit on train, transform both (never fit on test — that would be data leakage)
    X_train = vectorizer.fit_transform(train_strings)
    X_test  = vectorizer.transform(test_strings)

    print(f"[TF-IDF] Matrix: train={X_train.shape}, test={X_test.shape}")

    # Hand off to core which runs SVM / NB / LR
    train_and_evaluate("TF-IDF", X_train, X_test, y_train, y_test)
