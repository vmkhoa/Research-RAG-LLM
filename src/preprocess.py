import re
from typing import List
from nltk.tokenize import sent_tokenize


def remove_irrelevant_sections(text: str) -> str:
    """Removes irrelevant sections such as reference, acknowledgments."""
    pattern = r'\b(references|bibliography|acknowledg(e)?ments)\b'
    parts = re.split(pattern, text, flags=re.IGNORECASE)
    return parts[0].strip()

def remove_equations(text: str) -> str:
    """Removes LaTeX and replaces with the [EQUATION] token."""
    return re.sub(r'\$.*?\$', '[EQUATION]', text)

def normalize_text(text: str) -> str:
    """Removes non-ASCII characters and indentations or extra whitespaces and substitutes
    with a single whitespace."""
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def clean_text(text: str) -> str:
    """Cleans textual content by performing the above methods."""
    text = remove_irrelevant_sections(text)
    text = remove_equations(text)
    text = normalize_text(text)
    return text

def chunk_sentences(text: str, chunk_size: int = 5) -> List[str]:
    """Tokenizes textual content into sentences and groups them
    as chunks of 5."""
    sentences = sent_tokenize(text)
    return [' '.join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]
