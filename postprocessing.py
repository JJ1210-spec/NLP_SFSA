# =============================================================================
# postprocessing.py
# Two jobs:
#   1. Rule-based overrides — catch difficult feedbacks that models get wrong.
#   2. Aspect-wise sentiment breakdown — tell the user which sentiment was
#      expressed about teaching, exams, classroom, etc.
# =============================================================================

from preprocessing import detect_aspects


# ---------------------------------------------------------------------------
# ── Rule set ──
# Each rule is a dict with:
#   "keywords"  : list of words/phrases that must appear in the text
#   "sentiment" : the label to force when the rule fires
#   "reason"    : human-readable explanation (for debugging)
#
# Rules are checked in ORDER — the first match wins.
# Add more rules as you discover patterns from the confusion matrices.
# ---------------------------------------------------------------------------

OVERRIDE_RULES = [

    # ── Strong negation of positive words ───────────────────────────────────
    {
        "keywords": ["not good", "not great", "not helpful", "not clear",
                     "not satisfied", "not worth"],
        "sentiment": "negative",
        "reason": "Negated positive phrase → negative",
    },
    {
        "keywords": ["not bad", "not terrible", "not poor"],
        "sentiment": "positive",
        "reason": "Negated negative phrase → positive",
    },

    # ── Sarcasm / irony cues ─────────────────────────────────────────────────
    {
        "keywords": ["wow such a great", "oh very helpful", "yeah right",
                     "totally useful", "absolutely useless"],
        "sentiment": "negative",
        "reason": "Likely sarcasm → negative",
    },

    # ── Mixed / hedging language → neutral ───────────────────────────────────
    {
        "keywords": ["could be better", "room for improvement",
                     "okay but", "decent but", "average but",
                     "mixed feelings", "not sure", "somewhat"],
        "sentiment": "neutral",
        "reason": "Hedging language → neutral",
    },

    # ── Explicit extreme negative ─────────────────────────────────────────────
    {
        "keywords": ["worst", "terrible", "pathetic", "awful",
                     "waste of time", "completely useless", "absolutely horrible"],
        "sentiment": "negative",
        "reason": "Extreme negative keyword → negative",
    },

    # ── Explicit extreme positive ─────────────────────────────────────────────
    {
        "keywords": ["excellent", "outstanding", "superb", "fantastic",
                     "best course", "highly recommend", "loved it", "amazing"],
        "sentiment": "positive",
        "reason": "Extreme positive keyword → positive",
    },

    # ── Grievance about fairness (often wrongly predicted as neutral) ─────────
    {
        "keywords": ["not fair", "unfair grading", "biased", "partial marking",
                     "discriminated"],
        "sentiment": "negative",
        "reason": "Fairness grievance → negative",
    },

    # ── Polite but empty praise (often wrongly predicted as positive) ─────────
    {
        "keywords": ["nothing special", "just average", "nothing new",
                     "nothing much to say", "nothing stood out"],
        "sentiment": "neutral",
        "reason": "Polite but empty praise → neutral",
    },
]


# ---------------------------------------------------------------------------
# Apply overrides to a list of predicted labels
# ---------------------------------------------------------------------------

def apply_rules(texts: list, predictions: list) -> tuple:
    """
    Walks through each (text, predicted_label) pair.
    If a rule fires, the prediction is overridden.

    Returns
    -------
    corrected_predictions : list[str]
    override_log          : list[dict]  — records what was changed and why
    """
    corrected = []
    override_log = []

    for text, pred in zip(texts, predictions):
        text_lower = text.lower()
        fired_rule = None

        for rule in OVERRIDE_RULES:
            # Check if ANY keyword in the rule appears in the text
            if any(kw in text_lower for kw in rule["keywords"]):
                fired_rule = rule
                break   # first matching rule wins

        if fired_rule and fired_rule["sentiment"] != pred:
            # Rule overrides the model's prediction
            override_log.append({
                "text":        text,
                "original":    pred,
                "corrected":   fired_rule["sentiment"],
                "reason":      fired_rule["reason"],
            })
            corrected.append(fired_rule["sentiment"])
        else:
            corrected.append(pred)

    return corrected, override_log


# ---------------------------------------------------------------------------
# Aspect-wise sentiment summary
# ---------------------------------------------------------------------------

def aspect_sentiment_summary(texts: list, labels: list) -> dict:
    """
    For each aspect, count how many feedbacks mentioning that aspect
    were positive / negative / neutral.

    Parameters
    ----------
    texts  : list[str]  — raw feedback strings
    labels : list[str]  — predicted (or corrected) sentiment labels

    Returns
    -------
    summary : dict  { aspect: { "positive": n, "negative": n, "neutral": n } }
    """
    from config import CFG

    # Initialise counts
    summary = {
        aspect: {"positive": 0, "negative": 0, "neutral": 0}
        for aspect in list(CFG.ASPECTS.keys()) + ["general"]
    }

    for text, label in zip(texts, labels):
        aspects = detect_aspects(text.lower())
        for aspect in aspects:
            if aspect in summary:
                summary[aspect][label] = summary[aspect].get(label, 0) + 1

    return summary


# ---------------------------------------------------------------------------
# Print a readable aspect-wise report
# ---------------------------------------------------------------------------

def print_aspect_report(summary: dict):
    """Prints the aspect-sentiment summary as a formatted table."""

    print("\n" + "=" * 60)
    print("ASPECT-WISE SENTIMENT BREAKDOWN")
    print("=" * 60)
    print(f"{'Aspect':<20} {'Positive':>10} {'Negative':>10} {'Neutral':>10}  {'Total':>7}")
    print("-" * 60)

    for aspect, counts in summary.items():
        pos = counts.get("positive", 0)
        neg = counts.get("negative", 0)
        neu = counts.get("neutral",  0)
        total = pos + neg + neu
        if total == 0:
            continue    # skip aspects with no data
        print(f"{aspect:<20} {pos:>10} {neg:>10} {neu:>10}  {total:>7}")

    print("=" * 60)
