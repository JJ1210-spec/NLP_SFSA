# =============================================================================
# embeddings/bert.py
# BERT contextual embeddings using HuggingFace Transformers.
#
# Strategy:
#   Pass each feedback through BERT.
#   Take the [CLS] token vector from the last hidden state as the
#   sentence representation — this is the standard approach for
#   classification tasks.
# =============================================================================

import numpy as np
from core import train_and_evaluate
from config import CFG


def run_bert(
    train_strings: list,   # plain text strings (raw or lightly cleaned)
    test_strings: list,
    y_train: list,
    y_test: list,
):
    """
    Generates BERT [CLS] embeddings for every feedback string.
    Processes in batches to avoid OOM errors.
    Requires: pip install transformers torch
    """

    print(f"\n[BERT] Loading model '{CFG.BERT_MODEL}' ...")

    try:
        import torch
        from transformers import BertTokenizer, BertModel
    except ImportError:
        print("[BERT] SKIPPED — run: pip install transformers torch")
        return

    # --- Load tokeniser and model ---
    tokenizer = BertTokenizer.from_pretrained(CFG.BERT_MODEL)
    model     = BertModel.from_pretrained(CFG.BERT_MODEL)
    model.eval()   # disable dropout layers for deterministic output

    # Use GPU if available — speeds up encoding significantly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"[BERT] Using device: {device}")

    # --- Encode a list of strings in mini-batches ---
    def encode_batch(strings: list) -> np.ndarray:
        all_cls = []
        batch_size = CFG.BERT_BATCH

        for i in range(0, len(strings), batch_size):
            batch = strings[i : i + batch_size]

            # Tokenise: pad/truncate to CFG.BERT_MAX_LEN
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=CFG.BERT_MAX_LEN,
                return_tensors="pt",
            )

            # Move tensors to device
            input_ids      = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask)

            # last_hidden_state shape: (batch, seq_len, hidden_size)
            # Index 0 = [CLS] token representation
            cls_vectors = outputs.last_hidden_state[:, 0, :]
            all_cls.append(cls_vectors.cpu().numpy())

            if (i // batch_size) % 5 == 0:
                print(f"[BERT]   Encoded {min(i + batch_size, len(strings))}"
                      f" / {len(strings)}")

        return np.vstack(all_cls)

    X_train = encode_batch(train_strings)
    X_test  = encode_batch(test_strings)

    print(f"[BERT] Matrix: train={X_train.shape}, test={X_test.shape}")
    train_and_evaluate("BERT", X_train, X_test, y_train, y_test)
