"""
Text preprocessing for spam classification.
Cleans raw text and prepares it for TF-IDF vectorization.
"""
import re
import string


def clean_text(text: str) -> str:
    """
    Basic text cleaning pipeline:
    1. Lowercase everything
    2. Remove URLs, phone numbers, email addresses
    3. Remove punctuation and extra whitespace
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove phone numbers (7-15 digits with optional dashes/spaces and optional leading '+')
    text = re.sub(r"(?<!\w)\+?[\d\s\-]{7,15}(?!\w)", " ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_csv(path: str) -> tuple:
    """
    Load a TSV/CSV file with label in column 1, text in column 2.
    Handles the SMS Spam Collection format: ham/spam <tab> message

    Returns (texts, labels) where labels are 0=ham, 1=spam.
    """
    import csv

    texts, labels = [], []

    # Detect delimiter from first line
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline()

    if "\t" in first_line:
        delimiter = "\t"
    else:
        delimiter = ","

    # Heuristic: no header if first field is "ham" or "spam"
    first_field = first_line.split(delimiter)[0].strip().lower() if first_line else ""
    has_header = first_field not in ("ham", "spam")

    print(f"   Delimiter: {'TAB' if delimiter == chr(9) else 'comma'}")
    print(f"   Header: {has_header}  |  First field: {first_field!r}")

    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter=delimiter)
        if has_header:
            next(reader, None)

        for row in reader:
            if len(row) < 2:
                continue

            label_raw = row[0].strip().lower()
            text_raw = row[1].strip()

            # Normalize label
            label = 1 if label_raw == "spam" else 0

            if text_raw:
                texts.append(clean_text(text_raw))
                labels.append(label)

    return texts, labels
