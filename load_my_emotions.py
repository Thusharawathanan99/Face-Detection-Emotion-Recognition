"""Simple loader for `my_emotions.csv` that provides programmatic access
without any AI dependencies.

Usage:
    from load_my_emotions import load_emotions, get_description
    data = load_emotions()
    print(get_description('happy'))
"""
import csv
from pathlib import Path

BASE = Path(__file__).resolve().parent
CSV_PATH = BASE / 'my_emotions.csv'


def load_emotions(path: Path = None):
    """Load the CSV and return a dict label->description."""
    p = Path(path) if path else CSV_PATH
    d = {}
    if not p.exists():
        return d
    with p.open(newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            label = (r.get('label') or '').strip()
            desc = (r.get('description') or '').strip()
            if label:
                d[label] = desc
    return d


def get_description(label: str, data: dict = None) -> str:
    """Return description for label (case-insensitive)."""
    if data is None:
        data = load_emotions()
    if not label:
        return 'No description available.'
    # exact or case-insensitive match
    if label in data:
        return data[label]
    low = label.lower()
    for k, v in data.items():
        if k.lower() == low:
            return v
    return 'No description available.'


if __name__ == '__main__':
    print('Loaded', len(load_emotions()), 'emotions')