"""
Target languages for the safety representations study.

Chosen to test:
  - High-resource (English, French, Arabic) vs low-resource (Yoruba, Swahili)
  - Languages with and without dedicated Aya regional training
"""

# (ISO 639-1 code, display name, NLLB code, is_low_resource)
LANGUAGES = [
    ("en", "English",  "eng_Latn", False),
    ("fr", "French",   "fra_Latn", False),
    ("ar", "Arabic",   "arb_Arab", False),
    ("zh", "Chinese",  "zho_Hans", False),
    ("sw", "Swahili",  "swh_Latn", True),
    ("yo", "Yoruba",   "yor_Latn", True),
]

LANG_CODES   = [l[0] for l in LANGUAGES]
LANG_NAMES   = {l[0]: l[1] for l in LANGUAGES}
NLLB_CODES   = {l[0]: l[2] for l in LANGUAGES}
LOW_RESOURCE = {l[0]: l[3] for l in LANGUAGES}

# Prompts per language target
N_HARMFUL  = 10_000
N_HARMLESS = 10_000
