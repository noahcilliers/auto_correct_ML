"""
Training data for the autocorrect model: lists of (wrong, right) word pairs.

    from data import synthetic, norvig, batches, mixed_batches, grab, encode

    synth = synthetic(500_000)           # en_words.txt words with random edits
    train, val = norvig()                # real misspellings, split by correct word

    for batch in batches(synth, 64):     # one shuffled pass, lists of (wrong, right)
        src, tgt_in, tgt_out = encode(batch)

Rules from DATA_PLAN.md, applied to every pair: lowercase, `^[a-z][a-z']*[a-z]$`,
2-20 letters, the right side must be a dictionary word, the wrong side must not
be one (the editor never sends known words to the model, so such pairs would be
wasted). Pure Python, no numpy or torch, so it imports anywhere.
"""
import random
import re
from itertools import accumulate
from pathlib import Path

from spellcheck import FREQUENCIES, is_known

DATA = Path(__file__).resolve().parent / "data"
NORVIG = DATA / "norvig.txt"

WORD = re.compile(r"^[a-z][a-z']*[a-z]$")
MIN_LEN, MAX_LEN = 2, 20


def usable(w: str) -> bool:
    return MIN_LEN <= len(w) <= MAX_LEN and WORD.match(w) is not None


# The output vocabulary: every word the model may emit. Sampling weight is
# count ** 0.4, which flattens the Zipf curve so rare words still get seen.
VOCAB = [w for w in FREQUENCIES if usable(w)]
VOCAB_SET = set(VOCAB)
_CUM_WEIGHTS = list(accumulate(FREQUENCIES[w] ** 0.4 for w in VOCAB))


# ---------------------------------------------------------------- characters

PAD, SOS, EOS = 0, 1, 2
CHARS = "abcdefghijklmnopqrstuvwxyz'"
STOI = {c: i + 3 for i, c in enumerate(CHARS)}
ITOS = {i: c for c, i in STOI.items()}
VOCAB_SIZE = len(CHARS) + 3  # what Seq2Seq(vocab_size=...) wants


def encode(batch):
    """
    (wrong, right) pairs -> (src, tgt_in, tgt_out): lists of equal-length int
    lists, padded on the right with PAD. tgt_in is SOS + right and tgt_out is
    right + EOS, so the decoder predicts tgt_out[t] having seen tgt_in[:t + 1].
    Wrap each in torch.tensor(); a mask is simply `!= PAD`.
    """
    src = _pad([[STOI[c] for c in wrong] for wrong, _ in batch])
    right = [[STOI[c] for c in r] for _, r in batch]
    tgt_in = _pad([[SOS] + r for r in right])
    tgt_out = _pad([r + [EOS] for r in right])
    return src, tgt_in, tgt_out


def decode(ids) -> str:
    """Int ids back to a word. Stops at EOS, ignores PAD and SOS."""
    out = []
    for i in ids:
        i = int(i)
        if i == EOS:
            break
        if i in ITOS:
            out.append(ITOS[i])
    return "".join(out)


def _pad(rows):
    width = max(len(r) for r in rows)
    return [r + [PAD] * (width - len(r)) for r in rows]


# ----------------------------------------------------------------- synthetic

LETTERS = "abcdefghijklmnopqrstuvwxyz"


def _edit(w: str, rng: random.Random) -> str:
    """One random edit: insert, delete, substitute, transpose, double or undouble."""
    if len(w) < 2:
        return w
    op = rng.randrange(6)
    i = rng.randrange(len(w))
    if op == 0:  # insert
        j = rng.randrange(len(w) + 1)
        return w[:j] + rng.choice(LETTERS) + w[j:]
    if op == 1:  # delete
        return w[:i] + w[i + 1:]
    if op == 2:  # substitute
        return w[:i] + rng.choice(LETTERS) + w[i + 1:]
    if op == 3:  # transpose
        i = min(i, len(w) - 2)
        return w[:i] + w[i + 1] + w[i] + w[i + 2:]
    if op == 4:  # double
        return w[:i] + w[i] + w[i:]
    doubles = [k for k in range(len(w) - 1) if w[k] == w[k + 1]]  # undouble
    if not doubles:
        return w
    k = rng.choice(doubles)
    return w[:k] + w[k + 1:]


def corrupt(word: str, rng: random.Random, tries: int = 20) -> str | None:
    """
    A misspelling of `word` one edit away (75%) or two (25%) that the dictionary
    does not know. None if `tries` attempts all landed on real words, which
    happens for short words ("cat" -> "cot").
    """
    for _ in range(tries):
        w = _edit(word, rng)
        if rng.random() < 0.25:
            w = _edit(w, rng)
        if w != word and usable(w) and not is_known(w):
            return w
    return None


def synthetic(n: int = 500_000, seed: int = 0, clean_frac: float = 0.1) -> list[tuple[str, str]]:
    """
    n (wrong, right) pairs built from en_words.txt. Words are drawn with weight
    count ** 0.4 and corrupted; about `clean_frac` of them are left as-is so the
    model learns to copy a word through unchanged. Deterministic for a given
    seed, so pass a new seed each epoch if you want fresh corruptions.
    """
    rng = random.Random(seed)
    out: list[tuple[str, str]] = []
    while len(out) < n:
        for right in rng.choices(VOCAB, cum_weights=_CUM_WEIGHTS, k=n - len(out)):
            if rng.random() < clean_frac:
                out.append((right, right))
                continue
            wrong = corrupt(right, rng)
            if wrong is not None:
                out.append((wrong, right))
    return out


# -------------------------------------------------------------------- norvig

def norvig(val_frac: float = 0.2, seed: int = 0) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """
    Real misspellings from data/norvig.txt as (train, val) lists of (wrong, right).
    Split by correct word, so a word in val never appears in train in any form.
    Train repeats a pair as many times as Norvig counted it (`fore*5`); val keeps
    each distinct pair once, so accuracy is measured over distinct errors.
    """
    # right -> {wrong: count}. A dict because Norvig sometimes lists the same
    # misspelling twice on a line ("Look" and "look*2"); the counts are merged.
    by_word: dict[str, dict[str, int]] = {}
    with open(NORVIG, encoding="utf-8") as f:
        for line in f:
            right, sep, rest = line.strip().lower().partition(":")
            right = right.strip()
            if not sep or not usable(right) or right not in VOCAB_SET:
                continue
            for item in rest.split(","):
                wrong, _, count = item.strip().partition("*")
                wrong = wrong.strip()
                if wrong == right or not usable(wrong) or is_known(wrong):
                    continue
                errors = by_word.setdefault(right, {})
                errors[wrong] = errors.get(wrong, 0) + int(count or 1)

    words = sorted(by_word)
    random.Random(seed).shuffle(words)
    held_out = set(words[: round(len(words) * val_frac)])

    train: list[tuple[str, str]] = []
    val: list[tuple[str, str]] = []
    for right, errors in by_word.items():
        if right in held_out:
            val.extend((wrong, right) for wrong in errors)
        else:
            train.extend((wrong, right) for wrong, count in errors.items() for _ in range(count))
    return train, val


# ------------------------------------------------------------------- batches

def batches(pairs, size: int, seed=None):
    """One shuffled pass over `pairs`, yielding lists of `size` (the last may be short)."""
    order = list(range(len(pairs)))
    random.Random(seed).shuffle(order)
    for i in range(0, len(order), size):
        yield [pairs[j] for j in order[i:i + size]]


def grab(pairs, size: int, rng=random):
    """A single random batch of `size` distinct pairs."""
    return rng.sample(pairs, size)


def mixed_batches(synth, norvig_train, size: int, norvig_frac: float = 0.3, seed=None):
    """
    Phase-2 batches: one shuffled pass over `synth`, with each batch topped up to
    `size` by `norvig_frac` of Norvig pairs drawn at random (with replacement).
    """
    rng = random.Random(seed)
    n_norvig = round(size * norvig_frac)
    for chunk in batches(synth, size - n_norvig, seed=rng.random()):
        batch = chunk + rng.choices(norvig_train, k=n_norvig)
        rng.shuffle(batch)
        yield batch


if __name__ == "__main__":
    import time

    print(f"vocab: {len(VOCAB)} words, {VOCAB_SIZE} character ids")

    t = time.perf_counter()
    train, val = norvig()
    print(f"norvig: {len(train)} train pairs over {len({r for _, r in train})} words, "
          f"{len(val)} val pairs over {len({r for _, r in val})} words  ({time.perf_counter() - t:.1f}s)")
    print("  train:", train[:4])
    print("  val:  ", val[:4])

    t = time.perf_counter()
    synth = synthetic(500_000)
    print(f"synthetic: {len(synth)} pairs  ({time.perf_counter() - t:.1f}s)")
    print("  ", synth[:6])

    batch = grab(synth, 3)
    src, tgt_in, tgt_out = encode(batch)
    print("batch:  ", batch)
    print("src:    ", src)
    print("tgt_in: ", tgt_in)
    print("tgt_out:", tgt_out)
    print("decoded:", [decode(row) for row in tgt_out])
