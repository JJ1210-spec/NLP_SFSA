# =============================================================================
# main.py
# The single entry point that runs the entire pipeline.
#
# Flow:
#   1.  Load data
#   2.  Preprocess  (tokenised + clean strings)
#   3.  Run each embedding  →  each trains 3 classifiers  →  results appended
#   4.  Print comparison table + confusion matrices
#   5.  Identify best model
#   6.  Extract hard (misclassified) samples
#   7.  Retrain best model on hard samples
#   8.  Apply rule-based post-processing
#   9.  Print aspect-wise sentiment breakdown
# =============================================================================

import os

# ── Our modules ──────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split
from config import CFG
from data_loader import load_data

from features import (
    run_preprocess,
    preprocess_batch,
    get_meta_matrix,
    get_clean_strings,
    get_neg_strings,
)
from core import ALL_RESULTS
from embeddings.tfidf          import run_tfidf
from embeddings.static_embeddings import run_word2vec, run_glove, run_fasttext
from embeddings.sbert          import run_sbert
from embeddings.bert           import run_bert
from reporting import (
    print_comparison_table,
    plot_confusion_matrices,
    get_hard_samples,
    retrain_on_hard_samples,
)
from postprocessing import apply_rules, aspect_sentiment_summary, print_aspect_report


# =============================================================================
# STEP 1 — Load data
# =============================================================================
print("\n── STEP 1: Loading data ──────────────────────────────────────")
#train_texts, test_texts, train_labels, test_labels = load_data()
path = "C:\\Jaiyanth_elitebook_backup\\E drive\\Jaiyanth Jitendra\\sentiment_pipeline\\final_dataset.csv"

texts, labels, le = load_data(path)
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# =============================================================================
# STEP 2 — Preprocess
# =============================================================================
print("\n── STEP 2: Preprocessing ────────────────────────────────────")

# preprocess_batch returns list[ list[list[str]] ]
# e.g. for "The exam was not fair but good lecture" it returns
#   [  ["exam", "fair_NEG"],   ["good", "lecture"]  ]
train_preprocessed = preprocess_batch(train_texts)
test_preprocessed  = preprocess_batch(test_texts)

# Clean strings for TF-IDF, BERT, SBERT
train_clean = get_clean_strings(train_preprocessed)
test_clean  = get_clean_strings(test_preprocessed)

print(f"Example clean string: {train_clean[0]}")


# =============================================================================
# STEP 3 — Run embeddings + classifiers
# =============================================================================
print("\n── STEP 3: Embeddings + Classification ──────────────────────")

# -- TF-IDF (baseline, always fast) --
if CFG.RUN_TFIDF:
    run_tfidf(train_clean, test_clean, train_labels, test_labels)

# -- Word2Vec (trained on our corpus) --
if CFG.RUN_WORD2VEC:
    run_word2vec(train_preprocessed, test_preprocessed, train_labels, test_labels)

# -- GloVe (needs pre-trained file at CFG.GLOVE_PATH) --
if CFG.RUN_GLOVE:
    run_glove(train_preprocessed, test_preprocessed, train_labels, test_labels)

# -- FastText (needs pre-trained .bin at CFG.FASTTEXT_PATH) --
if CFG.RUN_FASTTEXT:
    run_fasttext(train_preprocessed, test_preprocessed, train_labels, test_labels)

# -- SBERT (downloads model automatically) --
if CFG.RUN_SBERT:
    run_sbert(train_clean, test_clean, train_labels, test_labels)

# -- BERT CLS embeddings --
if CFG.RUN_BERT:
    run_bert(train_clean, test_clean, train_labels, test_labels)


# =============================================================================
# STEP 4 — Comparison table + confusion matrices
# =============================================================================
print("\n── STEP 4: Results ──────────────────────────────────────────")

comparison_df = print_comparison_table(ALL_RESULTS)
plot_confusion_matrices(ALL_RESULTS)


# =============================================================================
# STEP 5 — Identify the best model
# =============================================================================
best_result = max(ALL_RESULTS, key=lambda r: r["accuracy"])
print(f"\nBest combo: {best_result['embedding']} + {best_result['classifier']}"
      f"  →  {best_result['accuracy']}%")


# =============================================================================
# STEP 6 — Extract hard (misclassified) samples from test set
# =============================================================================
print("\n── STEP 6: Hard sample extraction ───────────────────────────")
hard_texts, hard_labels = get_hard_samples(best_result, test_texts)


# =============================================================================
# STEP 7 — Retrain best model on original train + hard samples
#
# NOTE: We need the embedding vectors for both the original train set
# and the hard samples.  For simplicity we reuse TF-IDF here because
# it is always available.  If your best model is BERT/SBERT, replace
# the vectoriser call below with the appropriate encoder.
# =============================================================================
if len(hard_texts) > 0:
    print("\n── STEP 7: Retraining on hard samples ───────────────────────")

    from sklearn.feature_extraction.text import TfidfVectorizer

    # Re-vectorise with TF-IDF (same settings as in run_tfidf)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10_000,
                                 sublinear_tf=True)
    X_train_vec = vectorizer.fit_transform(train_clean)

    # Vectorise the hard samples using the SAME fitted vectoriser
    hard_clean  = get_clean_strings(preprocess_batch(hard_texts))
    X_hard_vec  = vectorizer.transform(hard_clean)

    X_test_vec  = vectorizer.transform(test_clean)

    # Find the TF-IDF SVM result (or best TF-IDF result) to retrain
    tfidf_results = [r for r in ALL_RESULTS if r["embedding"] == "TF-IDF"]
    if tfidf_results:
        best_tfidf = max(tfidf_results, key=lambda r: r["accuracy"])
        retrain_on_hard_samples(
            best_result   = best_tfidf,
            X_train_hard  = X_hard_vec.toarray(),
            y_train_hard  = hard_labels,
            X_train_orig  = X_train_vec.toarray(),
            y_train_orig  = train_labels,
            X_test        = X_test_vec.toarray(),
            y_test        = test_labels,
        )


# =============================================================================
# STEP 8 — Post-processing: rule-based overrides on test predictions
# =============================================================================
print("\n── STEP 8: Rule-based post-processing ───────────────────────")

raw_predictions = list(best_result["y_pred"])
corrected_predictions, override_log = apply_rules(test_texts, raw_predictions)

print(f"  Rules overrode {len(override_log)} prediction(s).")
for entry in override_log[:5]:   # show first 5 for brevity
    print(f"  TEXT     : {entry['text'][:70]}")
    print(f"  ORIGINAL : {entry['original']}  →  CORRECTED: {entry['corrected']}")
    print(f"  REASON   : {entry['reason']}\n")


# =============================================================================
# STEP 9 — Aspect-wise sentiment breakdown on test set
# =============================================================================
print("\n── STEP 9: Aspect-wise breakdown ────────────────────────────")
summary = aspect_sentiment_summary(test_texts, corrected_predictions)
print_aspect_report(summary)


print("\n✓ Pipeline complete.\n")
