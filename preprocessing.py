# =============================================================================
# preprocessing.py
# The full text-cleaning pipeline — updated from the original to also:
#   • detect which ASPECTS each feedback mentions
#   • return both token lists (for static embeddings) and
#     clean strings (for BERT / SBERT / TF-IDF)
# =============================================================================

import re
import string
from config import CFG

# ---------------------------------------------------------------------------
# Constants (same as original, kept here to make this file self-contained)
# ---------------------------------------------------------------------------

_REPEAT_PATTERN = re.compile(r"(.)\1{2,}")

_CLAUSE_KEYWORDS = ["but", "however", "although", "though", "yet", "while", "whereas"]
_CLAUSE_SPLIT_PATTERN = re.compile(
    r"\b(" + "|".join(_CLAUSE_KEYWORDS) + r")\b", re.IGNORECASE
)

_BASE_STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will",
    "just", "don", "should", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain",
    "aren", "couldn", "didn", "doesn", "hadn", "hasn", "haven", "isn", "ma",
    "mightn", "mustn", "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn",
}

_SENTIMENT_PRESERVE = {
    "not", "no", "never", "nor", "neither", "hardly", "barely", "scarcely",
    "but", "however", "although", "though", "yet",
    "good", "bad", "great", "poor", "excellent", "terrible", "worst", "best",
    "very", "too", "really", "quite", "more", "most", "less", "least",
}

_STOPWORDS = _BASE_STOPWORDS - _SENTIMENT_PRESERVE

_NEGATION_CUES = {
    "not", "no", "never", "nor", "neither", "hardly", "barely", "scarcely",
    "without", "nobody", "nothing", "nowhere",
    "isn't", "aren't", "wasn't", "weren't", "don't", "doesn't",
    "didn't", "won't", "wouldn't", "couldn't", "shouldn't", "can't", "cannot",
}

_NEGATION_TERMINATORS = {".", ",", "!", "?", ";", ":", "but", "however", "although", "yet"}

_EMOJI_DICT = {
    "😊": "smile",   "😡": "angry",  "😢": "sad",     "😍": "love",
    "👍": "good",    "👎": "bad",    "😅": "sweat_smile", "🙄": "roll_of_eyes",
    "😒": "unamused","😤": "triumph","🤦": "face_palm",
}

_IRREGULAR_CONTRACTIONS = {
    "won't": "will not", "can't": "cannot",  "shan't": "shall not",
    "ain't": "is not",   "don't": "do not",  "doesn't": "does not",
    "didn't": "did not", "isn't": "is not",  "aren't": "are not",
    "wasn't": "was not", "weren't": "were not", "haven't": "have not",
    "hasn't": "has not", "hadn't": "had not", "wouldn't": "would not",
    "shouldn't": "should not", "couldn't": "could not",
    "mightn't": "might not",   "mustn't": "must not", "needn't": "need not",
}

_CONTRACTION_PATTERNS = [
    (re.compile(r"n't\b"), " not"),
    (re.compile(r"'re\b"), " are"),
    (re.compile(r"'ve\b"), " have"),
    (re.compile(r"'ll\b"), " will"),
    (re.compile(r"'d\b"),  " would"),
    (re.compile(r"'m\b"),  " am"),
    (re.compile(r"'s\b"),  " is"),
]


# ---------------------------------------------------------------------------
# Individual cleaning steps (same as original)
# ---------------------------------------------------------------------------

def lowercase(text: str) -> str:
    return text.lower()

def handle_emojis(text: str, mode: str = "convert") -> str:
    if mode == "remove":
        for emj in _EMOJI_DICT:
            text = text.replace(emj, "")
        return text
    if mode == "convert":
        for emj, word in _EMOJI_DICT.items():
            text = text.replace(emj, f" {word} ")
        return text
    raise ValueError(f"mode must be 'convert' or 'remove', got '{mode}'")

def expand_contractions(text: str) -> str:
    if not text:
        return text
    text = text.lower()
    for contr, full in _IRREGULAR_CONTRACTIONS.items():
        text = text.replace(contr, full)
    for pattern, replacement in _CONTRACTION_PATTERNS:
        text = pattern.sub(replacement, text)
    return text

def remove_punctuation_and_special_chars(text: str) -> str:
    return re.sub(r"[^a-z\s]", " ", text)

def remove_numbers(text: str, preserve_context: bool = False) -> str:
    if preserve_context:
        return re.sub(r"\b\d+\b", " ", text)
    return re.sub(r"\d+", " ", text)

def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def split_clauses(text: str) -> list:
    clauses = _CLAUSE_SPLIT_PATTERN.split(text)
    cleaned = [p.strip() for p in clauses
               if p.strip() and p.strip().lower() not in _CLAUSE_KEYWORDS]
    return cleaned if cleaned else [text]

def tokenize(text: str) -> list:
    return text.split()

def remove_stopwords(tokens: list) -> list:
    return [t for t in tokens if t not in _STOPWORDS]

def handle_negation(tokens: list, suffix: str = "_NEG", window_size: int = 3) -> list:
    result = []
    negation_window = 0
    for token in tokens:
        if token in _NEGATION_CUES:
            negation_window = window_size
            result.append(token)
        elif token in _NEGATION_TERMINATORS:
            negation_window = 0
            result.append(token)
        else:
            if negation_window > 0:
                result.append(token + suffix)
                negation_window -= 1
            else:
                result.append(token)
    return result

def normalize_repeated_chars(tokens: list, max_repeat: int = 2) -> list:
    replacement = r"\1" * max_repeat
    return [_REPEAT_PATTERN.sub(replacement, t) for t in tokens]

def _lemmatize_word(word: str) -> str:
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    if word.endswith("ing") and len(word) > 4:
        if len(word) > 5 and word[-4] == word[-5]:
            return word[:-4]
        return word[:-3]
    if word.endswith("ed") and len(word) > 3:
        if word.endswith("ied"):
            return word[:-3] + "y"
        if len(word) > 4 and word[-3] == word[-4]:
            return word[:-3]
        if word.endswith(("ked", "ved", "ced")):
            return word[:-1]
        return word[:-2]
    return word

def lemmatize_tokens(tokens: list) -> list:
    suffix_marker = "_NEG"
    result = []
    for token in tokens:
        has_neg = token.endswith(suffix_marker)
        raw_word = token[:-4] if has_neg else token
        lemma = _lemmatize_word(raw_word)
        result.append(lemma + (suffix_marker if has_neg else ""))
    return result


# ---------------------------------------------------------------------------
# NEW: Aspect detection
# ---------------------------------------------------------------------------

def detect_aspects(text: str) -> list:
    """
    Given a raw (lowercased) feedback string, returns a list of aspect names
    whose keyword lists have at least one match in the text.

    Example
    -------
    "The exam was not fair and the professor explained well"
    → ["teaching", "exam"]
    """
    found = []
    for aspect, keywords in CFG.ASPECTS.items():
        # Check if any keyword appears in the text
        for kw in keywords:
            if kw in text:
                found.append(aspect)
                break   # one match is enough for this aspect
    return found if found else ["general"]


# ---------------------------------------------------------------------------
# Main preprocess function — returns token lists per clause
# ---------------------------------------------------------------------------

def preprocess(
    text: str,
    emoji_mode: str = None,
    preserve_numbers: bool = None,
    negation_suffix: str = None,
    max_char_repeat: int = None,
    lemmatize: bool = None,
) -> list:
    """
    Full pipeline on a single feedback string.
    Defaults are taken from CFG if not explicitly passed.

    Returns
    -------
    list[list[str]]
        One token list per clause detected in the feedback.
    """

    # Fall back to config defaults
    emoji_mode       = emoji_mode       or CFG.EMOJI_MODE
    preserve_numbers = preserve_numbers if preserve_numbers is not None else CFG.PRESERVE_NUMBERS
    negation_suffix  = negation_suffix  or CFG.NEGATION_SUFFIX
    max_char_repeat  = max_char_repeat  if max_char_repeat is not None else CFG.MAX_CHAR_REPEAT
    lemmatize        = lemmatize        if lemmatize        is not None else CFG.LEMMATIZE

    if not text or not isinstance(text, str):
        return [[]]

    text = lowercase(text)
    text = expand_contractions(text)
    text = handle_emojis(text, mode=emoji_mode)
    text = remove_punctuation_and_special_chars(text)
    text = remove_numbers(text, preserve_context=preserve_numbers)
    text = normalize_whitespace(text)

    clauses = split_clauses(text)
    processed_clauses = []

    for clause in clauses:
        tokens = tokenize(clause)
        tokens = remove_stopwords(tokens)
        tokens = handle_negation(tokens, suffix=negation_suffix)
        tokens = normalize_repeated_chars(tokens, max_repeat=max_char_repeat)
        if lemmatize:
            tokens = lemmatize_tokens(tokens)
        if tokens:
            processed_clauses.append(tokens)

    return processed_clauses if processed_clauses else [[]]
