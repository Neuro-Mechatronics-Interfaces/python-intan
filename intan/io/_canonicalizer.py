import re
import difflib

# 1) Define your official label set (CamelCase)
CANON = {
    "rest": "Rest",
    "handopen": "HandOpen",
    "handclose": "HandClose",
    "fingersopen": "FingersOpen",
    "fingersclose": "FingersClose",
    "indexflexion": "IndexFlexion",
    "indexextension": "IndexExtension",
    "thumbflexion": "ThumbFlexion",
    "thumbextension": "ThumbExtension",
    "middleflexion": "MiddleFlexion",
    "middleextension": "MiddleExtension",
    "ringflexion": "RingFlexion",
    "ringextension": "RingExtension",
    "pinkyflexion": "PinkyFlexion",
    "pinkyextension": "PinkyExtension",
    "twofingerpinch": "TwoFingerPinch",
    "threefingerchuck": "ThreeFingerChuck",
}

# 2) Add common synonyms / file-name aliases → canonical key (lowercase, no separators)
SYNONYMS = {
    "fingersopen": "handopen",          # if you want FingersOpen → HandOpen, change target to "fingersopen"
    "fingersclose": "handclose",        # likewise, set to "fingersclose" if that’s your canonical choice
    "resting": "rest",
    "null": "rest",
    "nan": "rest",                      # treat missing labels as Rest; change to "unknown" if you prefer
    "index": "indexflexion",
    "index2": "indexflexion",
    "2fingerinch": "twofingerpinch",
    "3fingerchuck": "threefingerchuck",
    # typos that show up:
    "pinkyfelxion": "pinkyflexion",
    "pin kyflexion": "pinkyflexion",
    "pinkyflexon": "pinkyflexion",
    # the one you mentioned (extra space + odd casing is already handled by normalization)
}

def _normalize_token(s: str) -> str:
    """strip, remove separators, lowercase."""
    return re.sub(r"[\s_-]+", "", (s or "").strip()).lower()

def canonical_label(
    s: str,
    *,
    style: str = "lower",     # "camel" -> CamelCase name from CANON; "lower" -> canonical key
    allow_fuzzy: bool = True,
    fuzzy_cutoff: float = 0.88,
    strict: bool = False,     # if True, raise on unknown
) -> str:
    """
    Canonicalize a raw label string to your official namespace.

    - style="camel": return CamelCase (e.g., "PinkyFlexion")
    - style="lower": return canonical token key (e.g., "pinkyflexion")
    """
    tok = _normalize_token(s)

    # map synonyms (including “nan”→rest if you want that behavior)
    tok = SYNONYMS.get(tok, tok)

    if tok in CANON:
        return CANON[tok] if style == "camel" else tok

    if allow_fuzzy:
        # try fuzzy against known keys
        match = difflib.get_close_matches(tok, CANON.keys(), n=1, cutoff=fuzzy_cutoff)
        if match:
            m = match[0]
            return CANON[m] if style == "camel" else m

    if strict:
        raise ValueError(f"Unknown label: {s!r} → {tok!r}")

    # last resort: make a readable CamelCase from the normalized token
    # (still stable, but not guaranteed to be one of your official classes)
    fallback_camel = re.sub(r"([a-z])([0-9])", r"\1 \2", tok).title().replace(" ", "")
    return fallback_camel if style == "camel" else tok
