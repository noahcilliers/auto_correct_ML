"""
The autocorrect hook. This is the only thing the editor imports.

The editor calls correct() every time you finish typing a word, i.e. the
moment you type a space, newline or punctuation mark right after it.

    word     the word you just typed, e.g. "teh"
    context  everything in the document before that word
             (handy for a language model; ignore it if you don't need it)

Return the corrected word, or None to leave it alone.

Words the dictionary already knows never reach the model: correct() bails
out early via spellcheck.is_misspelled(), so the expensive part only runs
on words that are actually wrong.

The model is the character seq2seq from model.py. It is loaded once here,
at import, from BEST_MODEL next to this file, because correct() runs on the
UI thread and must stay well under ~50 ms per word.
"""
from pathlib import Path

import torch

from data import STOI, decode, usable, VOCAB_SIZE
from model import Seq2Seq, CHAR_EMB_DIM, CHECKPOINT
from spellcheck import is_known, is_misspelled, normalize

_DEVICE = "cpu"
_model = Seq2Seq(VOCAB_SIZE, CHAR_EMB_DIM, 256).to(_DEVICE)   # same shape as train()/test()
_model.load_state_dict(torch.load(Path(__file__).resolve().parent / CHECKPOINT,
                                  map_location=_DEVICE))
_model.eval()  # dropout off


def predict(word: str) -> str:
    """The model's guess for a lowercase a-z word: decode(generate(encode(word)))."""
    src = torch.tensor([[STOI[c] for c in word]], device=_DEVICE)
    return decode(_model.generate(src)[0])


def correct(word: str, context: str) -> str | None:
    if not is_misspelled(word, context):
        return None  # spelt fine (or a name / acronym / code); leave it alone

    w = normalize(word).lower()
    if not usable(w):
        return None  # not plain a-z, or too long: nothing the model was trained on

    fixed = predict(w)
    # The inference guard from DATA_PLAN.md: only ever show a dictionary word.
    # This also swallows the model's copy-through misses, since `w` isn't one.
    if not is_known(fixed):
        return None
    if word[0].isupper():
        fixed = fixed.capitalize()
    return fixed