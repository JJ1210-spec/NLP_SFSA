# =============================================================================
# features.py
# Helper functions that turn raw text lists into the forms every embedding
# module needs.
#
# Exported API (imported in main.py):
#   run_preprocess   – preprocess a single text, return list-of-token-lists
#   preprocess_batch – preprocess an entire list of texts
#   get_clean_strings– flat joined strings (for TF-IDF / BERT / SBERT)
#   get_neg_strings  – flat joined strings that preserve _NEG tags
#   get_meta_matrix  – metadata features: aspects, clause count, text length
# =============================================================================

import numpy as np
from preprocessing import preprocess, detect_aspects
from config import CFG


# ---------------------------------------------------------------------------
# Single text preprocessing
# ---------------------------------------------------------------------------

def run_preprocess(text: str) -> list:
    """
    Preprocess one feedback string.
    Returns list[list[str]] — one token list per clause.
    """
    return preprocess(text)


# ---------------------------------------------------------------------------
# Batch preprocessing
# ---------------------------------------------------------------------------

def preprocess_batch(texts: list) -> list:
    """
    Preprocess a list of raw feedback strings.
    Returns list[ list[list[str]] ] — one entry per text.
    """
    return [preprocess(t) for t in texts]


# ---------------------------------------------------------------------------
# Convert token-clause-lists → single flat string (no _NEG tags)
# Used by: TF-IDF, BERT, SBERT (they handle negation internally via context)
# ---------------------------------------------------------------------------

def get_clean_strings(preprocessed_batch: list) -> list:
    """
    Takes the output of preprocess_batch and joins all clause-tokens into
    one plain string per text.  _NEG suffixes are stripped.

    Example
    -------
    [[["good", "lecture"], ["not_NEG", "fair_NEG"]]]
    →  "good lecture not fair"
    """
    clean = []
    for clauses in preprocessed_batch:
        tokens = []
        for clause in clauses:
            for token in clause:
                # Strip the _NEG suffix for models that don't need it
                tokens.append(token.replace("_NEG", ""))
        clean.append(" ".join(tokens))
    return clean


# ---------------------------------------------------------------------------
# Flat strings WITH _NEG preserved
# Used by: Word2Vec, GloVe, FastText (where negation marking matters)
# ---------------------------------------------------------------------------

def get_neg_strings(preprocessed_batch: list) -> list:
    """
    Same as get_clean_strings but keeps _NEG suffixes.
    """
    neg_strings = []
    for clauses in preprocessed_batch:
        tokens = []
        for clause in clauses:
            tokens.extend(clause)
        neg_strings.append(" ".join(tokens))
    return neg_strings


# ---------------------------------------------------------------------------
# NEW: Metadata / aspect feature matrix
# Adds interpretable features on top of embeddings if desired.
# ---------------------------------------------------------------------------

def get_meta_matrix(texts: list, preprocessed_batch: list) -> np.ndarray:
    """
    Builds a small numeric feature matrix from metadata about each feedback:
      - One binary column per aspect  (1 if aspect mentioned, else 0)
      - clause_count  : how many clauses the text splits into
      - token_count   : total token count after preprocessing
      - avg_token_len : average character length of tokens

    Shape: (n_samples, n_aspects + 3)

    This matrix can be horizontally stacked with any embedding matrix to
    give classifiers extra signals.
    """
    aspect_names = list(CFG.ASPECTS.keys())
    rows = []

    for raw_text, clauses in zip(texts, preprocessed_batch):
        # --- Aspect binary flags ---
        detected = detect_aspects(raw_text.lower())
        aspect_flags = [1 if a in detected else 0 for a in aspect_names]

        # --- Structural features ---
        all_tokens = [tok for clause in clauses for tok in clause]
        clause_count  = len(clauses)
        token_count   = len(all_tokens)
        avg_token_len = (
            sum(len(t.replace("_NEG", "")) for t in all_tokens) / token_count
            if token_count > 0 else 0
        )

        rows.append(aspect_flags + [clause_count, token_count, avg_token_len])

    return np.array(rows, dtype=np.float32)
