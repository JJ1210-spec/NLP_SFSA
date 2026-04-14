# =============================================================================
# embeddings/static_embeddings.py
# Word2Vec, GloVe, FastText — all "static" (context-free) embeddings.
#
# Strategy for sentence-level vectors:
#   Average the word vectors for all tokens in the feedback.
#   Words not found in the vocabulary are skipped.
#   If no word was found at all, we use a zero vector.
# =============================================================================

import os
import numpy as np
from gensim.models import Word2Vec, KeyedVectors, FastText
from core import train_and_evaluate
from config import CFG


# ---------------------------------------------------------------------------
# Shared helper: average word vectors for a list of token-lists
# ---------------------------------------------------------------------------

def _average_vectors(tokenized_texts: list, wv, vector_size: int) -> np.ndarray:
    """
    For each text, look up each token in the word-vector model (wv)
    and return the average of the found vectors.

    Parameters
    ----------
    tokenized_texts : list[list[str]]  — one flat token list per feedback
    wv              : gensim KeyedVectors (or similar .get_vector() interface)
    vector_size     : int — dimensionality of the vectors

    Returns
    -------
    np.ndarray  shape (n_texts, vector_size)
    """
    matrix = []
    for tokens in tokenized_texts:
        vecs = []
        for token in tokens:
            # Try the token as-is; if _NEG is present, also try the base word
            candidates = [token, token.replace("_NEG", "")]
            for cand in candidates:
                try:
                    vecs.append(wv.get_vector(cand))
                    break   # found it, stop trying variants
                except KeyError:
                    continue

        if vecs:
            matrix.append(np.mean(vecs, axis=0))
        else:
            # No token was in the vocabulary — use zeros
            matrix.append(np.zeros(vector_size))

    return np.array(matrix, dtype=np.float32)


# ---------------------------------------------------------------------------
# Flatten clause-token-lists to a single token list per text
# ---------------------------------------------------------------------------

def _flatten(preprocessed_batch: list) -> list:
    """Convert list[list[list[str]]] → list[list[str]] (one list per text)."""
    return [[tok for clause in clauses for tok in clause]
            for clauses in preprocessed_batch]


# ===========================================================================
# Word2Vec — trained on OUR corpus (no external file needed)
# ===========================================================================

def run_word2vec(
    train_preprocessed: list,   # output of preprocess_batch() for train set
    test_preprocessed: list,
    y_train: list,
    y_test: list,
):
    """
    Trains a Word2Vec model on the training corpus, then averages vectors
    to get one embedding per feedback.
    """

    print("\n[Word2Vec] Training on corpus ...")

    train_tokens = _flatten(train_preprocessed)   # list of token lists
    test_tokens  = _flatten(test_preprocessed)

    # Train a Word2Vec model (Skip-gram by default with sg=1)
    model = Word2Vec(
        sentences=train_tokens,
        vector_size=CFG.W2V_VECTOR_SIZE,
        window=CFG.W2V_WINDOW,
        min_count=CFG.W2V_MIN_COUNT,
        epochs=CFG.W2V_EPOCHS,
        sg=1,   # 1 = Skip-gram, 0 = CBOW
        workers=4,
    )

    wv = model.wv   # KeyedVectors object

    # Average vectors for each feedback
    X_train = _average_vectors(train_tokens, wv, CFG.W2V_VECTOR_SIZE)
    X_test  = _average_vectors(test_tokens,  wv, CFG.W2V_VECTOR_SIZE)

    print(f"[Word2Vec] Matrix: train={X_train.shape}, test={X_test.shape}")
    train_and_evaluate("Word2Vec", X_train, X_test, y_train, y_test)


# ===========================================================================
# GloVe — loaded from a pre-trained .txt file (Stanford GloVe)
# ===========================================================================

def _load_glove(glove_path: str) -> tuple:
    """
    Reads a GloVe .txt file into a simple word→vector dictionary.

    Returns
    -------
    (word_to_vec: dict, vector_size: int)
    """
    word_to_vec = {}
    vector_size = None

    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            word  = parts[0]
            vec   = np.array(parts[1:], dtype=np.float32)
            word_to_vec[word] = vec
            if vector_size is None:
                vector_size = len(vec)

    print(f"[GloVe] Loaded {len(word_to_vec):,} vectors of size {vector_size}")
    return word_to_vec, vector_size


class _GloVeWrapper:
    """Thin wrapper so _average_vectors() can call .get_vector() on it."""
    def __init__(self, word_to_vec):
        self.wv = word_to_vec

    def get_vector(self, word: str):
        if word in self.wv:
            return self.wv[word]
        raise KeyError(word)


def run_glove(
    train_preprocessed: list,
    test_preprocessed: list,
    y_train: list,
    y_test: list,
):
    """Loads pre-trained GloVe vectors and averages them per feedback."""

    if not os.path.exists(CFG.GLOVE_PATH):
        print(f"[GloVe] SKIPPED — file not found: {CFG.GLOVE_PATH}")
        return

    print(f"\n[GloVe] Loading from {CFG.GLOVE_PATH} ...")
    word_to_vec, vector_size = _load_glove(CFG.GLOVE_PATH)
    wrapper = _GloVeWrapper(word_to_vec)

    train_tokens = _flatten(train_preprocessed)
    test_tokens  = _flatten(test_preprocessed)

    X_train = _average_vectors(train_tokens, wrapper, vector_size)
    X_test  = _average_vectors(test_tokens,  wrapper, vector_size)

    print(f"[GloVe] Matrix: train={X_train.shape}, test={X_test.shape}")
    train_and_evaluate("GloVe", X_train, X_test, y_train, y_test)


# ===========================================================================
# FastText — loads a pre-trained binary .bin model
# FastText handles OOV words by using sub-word (character n-gram) vectors.
# ===========================================================================

def run_fasttext(
    train_preprocessed: list,
    test_preprocessed: list,
    y_train: list,
    y_test: list,
):
    """Loads a pre-trained FastText .bin model and averages vectors."""

    if not os.path.exists(CFG.FASTTEXT_PATH):
        print(f"[FastText] SKIPPED — file not found: {CFG.FASTTEXT_PATH}")
        return

    print(f"\n[FastText] Loading from {CFG.FASTTEXT_PATH} ...")
    # gensim can load FastText's binary .bin files
    ft_model = FastText.load_fasttext_format(CFG.FASTTEXT_PATH)
    wv = ft_model.wv
    vector_size = ft_model.vector_size

    train_tokens = _flatten(train_preprocessed)
    test_tokens  = _flatten(test_preprocessed)

    X_train = _average_vectors(train_tokens, wv, vector_size)
    X_test  = _average_vectors(test_tokens,  wv, vector_size)

    print(f"[FastText] Matrix: train={X_train.shape}, test={X_test.shape}")
    train_and_evaluate("FastText", X_train, X_test, y_train, y_test)
