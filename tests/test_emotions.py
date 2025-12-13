import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import emotions
from load_my_emotions import load_emotions, get_description


def test_emotions_module():
    assert 'Joy' in emotions.EMOTION_DESCRIPTIONS
    assert emotions.get_description('joy').startswith('A bright')


def test_aliases():
    # aliases should normalize
    assert emotions.get_description('happy').startswith('A bright')
    assert emotions.get_description('neutral').startswith('A peaceful')


def test_csv_loader():
    data = load_emotions()
    assert 'Joy' in data
    assert get_description('joy', data).startswith('A bright')
