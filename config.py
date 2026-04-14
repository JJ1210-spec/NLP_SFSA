# =============================================================================
# config.py
# Central configuration for the entire pipeline.
# Change settings here — no need to touch any other file.
# =============================================================================

class CFG:
    # -------------------------------------------------------------------------
    # DATA
    # -------------------------------------------------------------------------
    DATA_PATH       = "data/feedback_data.csv"   # CSV with 'text' and 'label' columns
    TEXT_COL = "feedback_text"     # column name in your CSV
    LABEL_COL = "sentiment"   # column name in your CSV                   # Values: positive / negative / neutral
    TEST_SIZE       = 0.2                        # 20% held-out test set
    RANDOM_STATE    = 42

    # -------------------------------------------------------------------------
    # LABELS
    # -------------------------------------------------------------------------
    LABELS          = ["positive", "negative", "neutral"]

    # -------------------------------------------------------------------------
    # ASPECTS  — the facets we want to tag and analyse in each feedback
    # A feedback can touch one or many aspects.
    # -------------------------------------------------------------------------
    ASPECTS = {
        "teaching":       ["teach", "explain", "lecture", "professor", "instructor",
                           "faculty", "lesson", "presentation", "clarity", "engaging",
                           "boring", "interactive", "demo", "illustration"],

        "exam":           ["exam", "test", "quiz", "assessment", "question", "marks",
                           "grade", "score", "difficult", "easy", "fair", "unfair",
                           "paper", "mcq", "evaluation"],

        "assignments":    ["assignment", "homework", "project", "task", "deadline",
                           "submission", "workload", "practical", "lab", "report"],

        "curriculum":     ["syllabus", "curriculum", "content", "topic", "chapter",
                           "subject", "material", "coverage", "outdated", "relevant",
                           "course", "module", "unit"],

        "classroom":      ["classroom", "class", "room", "board", "projector", "seating",
                           "space", "facility", "infra", "infrastructure", "noisy",
                           "ventilation", "clean"],

        "online_resources": ["slide", "notes", "pdf", "recording", "video", "portal",
                             "upload", "lms", "moodle", "online", "resource", "material",
                             "access", "download"],

        "support":        ["doubt", "help", "office hours", "support", "responsive",
                           "approachable", "available", "feedback", "mentor", "guide",
                           "respond", "email", "staff"],

        "pace":           ["pace", "speed", "slow", "fast", "rush", "time", "duration",
                           "schedule", "timing", "late", "early", "manage"],
    }

    # -------------------------------------------------------------------------
    # PREPROCESSING
    # -------------------------------------------------------------------------
    EMOJI_MODE          = "convert"   # "convert" or "remove"
    PRESERVE_NUMBERS    = False
    NEGATION_SUFFIX     = "_NEG"
    MAX_CHAR_REPEAT     = 2
    LEMMATIZE           = True

    # -------------------------------------------------------------------------
    # EMBEDDINGS  — toggle which embeddings to run
    # -------------------------------------------------------------------------
    RUN_TFIDF    = True
    RUN_WORD2VEC = True
    RUN_GLOVE    = True
    RUN_FASTTEXT = True
    RUN_SBERT    = True
    RUN_BERT     = True

    # Paths to pre-trained model files (set these to your local paths)
    GLOVE_PATH    = "models/glove.6B.100d.txt"        # Download from Stanford
    FASTTEXT_PATH = "models/cc.en.300.bin"            # Download from fasttext.cc
    WORD2VEC_PATH = "models/GoogleNews-vectors-negative300.bin"  # Google News w2v

    # SBERT model name (downloads automatically via sentence-transformers)
    SBERT_MODEL   = "all-MiniLM-L6-v2"

    # BERT model name (downloads automatically via transformers)
    BERT_MODEL    = "bert-base-uncased"
    BERT_MAX_LEN  = 128
    BERT_BATCH    = 32

    # Word2Vec training settings (used when training on our own corpus)
    W2V_VECTOR_SIZE = 100
    W2V_WINDOW      = 5
    W2V_MIN_COUNT   = 1
    W2V_EPOCHS      = 10

    # -------------------------------------------------------------------------
    # CLASSIFIERS — all three run for every embedding
    # -------------------------------------------------------------------------
    CLASSIFIERS = ["svm", "nb", "lr"]          # support-vector, naïve bayes, logistic

    # -------------------------------------------------------------------------
    # OUTPUT / REPORTING
    # -------------------------------------------------------------------------
    RESULTS_DIR  = "results"
    FIGURES_DIR  = "results/figures"
