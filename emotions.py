"""Emotion labels and human-readable descriptions.

This module centralizes emotion names and the short descriptions you provided.
It is deliberately small and dependency-free so other modules can import it
without side effects.
"""

EMOTION_DESCRIPTIONS = {
    "Joy": "A bright, expansive feeling of pure delight and pleasure.",
    "Sadness": "A heavy, aching feeling of loss, disappointment, or sorrow.",
    "Anger": "A hot, sharp feeling of frustration, injustice, or being wronged.",
    "Fear": "A cold, alert feeling of dread, worry, or anticipation of threat.",
    "Love": "A warm, binding feeling of deep affection, attachment, and care.",
    "Disgust": "A strong feeling of revulsion or disapproval, whether physical or moral.",
    "Surprise": "A sudden, brief feeling of being startled or amazed by the unexpected.",
    "Shame": "A painful, shrinking feeling of humiliation or unworthiness.",
    "Guilt": "A heavy feeling of responsibility or regret for a wrong action.",
    "Pride": "A swelling, confident feeling of satisfaction in an achievement.",
    "Envy": "A bitter, yearning feeling of wanting what someone else has.",
    "Jealousy": "A fearful, possessive feeling over a threatened relationship.",
    "Grief": "A deep, profound sorrow, often following a loss; sadness amplified.",
    "Hope": "A light, optimistic feeling of expectation and desire for a positive future.",
    "Loneliness": "A hollow, aching feeling of isolation and longing for connection.",
    "Gratitude": "A full, warm feeling of thankfulness and appreciation.",
    "Anxiety": "A tense, knotted feeling of nervousness and apprehension about the future.",
    "Contentment": "A peaceful, calm feeling of quiet satisfaction and ease.",
    "Nostalgia": "A bittersweet, longing feeling for a fondly remembered past.",
    "Awe": "An overwhelming feeling of wonder and reverence, often in nature or art."
}


def get_description(label: str) -> str:
    """Return a best-effort description for the given emotion label.

    Matching is case-insensitive and will return a fallback string when the
    label is unknown.
    """
    if not label:
        return "No description available."

    canonical = normalize_label(label)
    if canonical and canonical in EMOTION_DESCRIPTIONS:
        return EMOTION_DESCRIPTIONS[canonical]

    return "No description available."


def all_emotions() -> list:
    """Return a list of supported emotion labels (stable ordering)."""
    return list(EMOTION_DESCRIPTIONS.keys())


# Common aliases and dataset label mappings -> canonical emotion labels
# Keys are lowercase expected input labels, values are the canonical label names
# defined in EMOTION_DESCRIPTIONS.
ALIASES = {
    # common simple mappings
    "happy": "Joy",
    "joyful": "Joy",
    "joy": "Joy",
    "sad": "Sadness",
    "angry": "Anger",
    "fearful": "Fear",
    "afraid": "Fear",
    "scared": "Fear",
    "love": "Love",
    "disgusted": "Disgust",
    "disgusting": "Disgust",
    "surprised": "Surprise",
    "shame": "Shame",
    "guilty": "Guilt",
    "proud": "Pride",
    "envy": "Envy",
    "jealous": "Jealousy",
    "grief": "Grief",
    "grieving": "Grief",
    "hopeful": "Hope",
    "lonely": "Loneliness",
    "thankful": "Gratitude",
    "anxious": "Anxiety",
    "content": "Contentment",
    "contentment": "Contentment",
    "nostalgic": "Nostalgia",
    "awe": "Awe",
    # legacy/neutral mapping
    "neutral": "Contentment",
}


def normalize_label(label: str) -> str:
    """Normalize a label to the canonical emotion label.

    - Handles case-insensitive matches against canonical labels.
    - Resolves known aliases (dataset folder names, adjective forms).
    - Returns the canonical label string if found, otherwise returns the
      original label (as provided) if it exactly matches a canonical key,
      or None when input is falsy.
    """
    if not label:
        return None

    # Quick check: exact canonical match (case-insensitive)
    for k in EMOTION_DESCRIPTIONS.keys():
        if k.lower() == label.lower():
            return k

    # Check aliases mapping
    mapped = ALIASES.get(label.lower())
    if mapped:
        return mapped

    # No normalization found
    return None
