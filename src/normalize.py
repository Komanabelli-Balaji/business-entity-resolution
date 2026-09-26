import re
import unicodedata

def normalize_text(text):
    if not isinstance(text, str) or not text: return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_prefix(text, n=4):
    if not text: return ""
    return text[:n]

def tokenize(text):
    if not text: return []
    return text.split()
