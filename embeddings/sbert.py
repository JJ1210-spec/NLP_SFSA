# =============================================================================
# embeddings/sbert.py
# Sentence-level embeddings: SBERT and InferSent.
#
# Both produce ONE vector per sentence directly — no averaging needed.
# They operate on clean raw strings, not on tokenised lists.
# =============================================================================

import os
import numpy as np
from core import train_and_evaluate
from config import CFG


# ===========================================================================
# SBERT  (sentence-transformers library — pip install sentence-transformers)
# ===========================================================================

def run_sbert(
    train_strings: list,   # plain text strings from get_clean_strings()
    test_strings: list,
    y_train: list,
    y_test: list,
):
    """
    Encodes each feedback as a single dense vector using a pre-trained
    SBERT model (default: all-MiniLM-L6-v2, 384-dim).
    The model downloads automatically on first use.
    """

    print(f"\n[SBERT] Loading model '{CFG.SBERT_MODEL}' ...")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("[SBERT] SKIPPED — run: pip install sentence-transformers")
        return

    model = SentenceTransformer(CFG.SBERT_MODEL)

    # encode() handles batching internally; show_progress_bar gives a nice bar
    X_train = model.encode(train_strings, show_progress_bar=True,
                            batch_size=64, convert_to_numpy=True)
    X_test  = model.encode(test_strings,  show_progress_bar=True,
                            batch_size=64, convert_to_numpy=True)

    print(f"[SBERT] Matrix: train={X_train.shape}, test={X_test.shape}")
    train_and_evaluate("SBERT", X_train, X_test, y_train, y_test)


# ===========================================================================
# InferSent  (Facebook Research — needs manual setup)
#
# Setup steps (one-time):
#   git clone https://github.com/facebookresearch/InferSent
#   pip install nltk
#   Download infersent2.pkl from the InferSent repo releases
#   Download fastText word vectors: crawl-300d-2M.vec.zip from fasttext.cc
# ===========================================================================

def run_infersent(
    train_strings: list,
    test_strings: list,
    y_train: list,
    y_test: list,
    infersent_model_path: str = "models/infersent2.pkl",
    w2v_path: str            = "models/crawl-300d-2M.vec",
    infersent_repo: str      = "InferSent",      # path to cloned repo
):
    """
    Encodes feedbacks using InferSent v2.
    Falls back gracefully if the model files are missing.
    """

    import sys

    # Check all required files exist before importing
    for path in [infersent_model_path, w2v_path]:
        if not os.path.exists(path):
            print(f"[InferSent] SKIPPED — file not found: {path}")
            return

    if not os.path.isdir(infersent_repo):
        print(f"[InferSent] SKIPPED — repo not found at: {infersent_repo}")
        return

    # Add InferSent repo to Python path so we can import its models module
    sys.path.insert(0, infersent_repo)

    try:
        import torch
        from models import InferSent   # from the cloned InferSent repo
    except ImportError as e:
        print(f"[InferSent] SKIPPED — import error: {e}")
        return

    print(f"\n[InferSent] Loading model from {infersent_model_path} ...")

    params_model = {
        "bsize": 64,
        "word_emb_dim": 300,
        "enc_lstm_dim": 2048,
        "pool_type": "max",
        "dpout_model": 0.0,
        "version": 2,   # InferSent version 2
    }

    model = InferSent(params_model)
    model.load_state_dict(torch.load(infersent_model_path, map_location="cpu"))
    model.set_w2v_path(w2v_path)
    model.build_vocab(train_strings, tokenize=True)

    X_train = model.encode(train_strings, tokenize=True, verbose=False)
    X_test  = model.encode(test_strings,  tokenize=True, verbose=False)

    X_train = np.array(X_train, dtype=np.float32)
    X_test  = np.array(X_test,  dtype=np.float32)

    print(f"[InferSent] Matrix: train={X_train.shape}, test={X_test.shape}")
    train_and_evaluate("InferSent", X_train, X_test, y_train, y_test)
