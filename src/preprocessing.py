import re
import string

def clean_text(text: str) -> str:

    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    text = re.sub(r"(?<!\w)\+?[\d\s\-]{7,15}(?!\w)", " ", text)

    text = re.sub(r"\S+@\S+", " ", text)

    text = text.translate(str.maketrans("", "", string.punctuation))

    text = re.sub(r"\s+", " ", text).strip()

    return text

def load_csv(path: str) -> tuple:

    import csv

    texts, labels = [], []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline()

    if "\t" in first_line:
        delimiter = "\t"
    else:
        delimiter = ","

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

            label = 1 if label_raw == "spam" else 0

            if text_raw:
                texts.append(clean_text(text_raw))
                labels.append(label)

    return texts, labels
