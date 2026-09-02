"""
Dictionary-based spell check.

This module answers one question: is this word spelt wrong? It never tries
to fix anything. Fixing is autocorrect.py's job, and it only gets called for
words that fail the check here, so the model never sees correctly spelt text.

Wordlist: data/en_words.txt, the SymSpell English frequency dictionary
(~83k words with corpus counts, MIT licence, github.com/wolfgarbe/SymSpell).
Counts are kept around in FREQUENCIES because they make a good prior when
ranking candidate corrections later.

Personal words: data/personal_words.txt, one word per line, is unioned in if
it exists. add_word() appends to it.
"""
import re
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
WORDLIST = DATA / "en_words.txt"
PERSONAL = DATA / "personal_words.txt"


def _load_frequencies(path: Path) -> dict[str, int]:
    freqs: dict[str, int] = {}
    with open(path, encoding="utf-8-sig") as f:  # -sig eats the BOM at the top
        for line in f:
            parts = line.split()
            if not parts:
                continue
            word = parts[0].lower()
            count = int(parts[1]) if len(parts) > 1 else 1
            freqs[word] = max(count, freqs.get(word, 0))
    return freqs


FREQUENCIES = _load_frequencies(WORDLIST)
WORDS = set(FREQUENCIES)
if PERSONAL.exists():
    WORDS |= {w.strip().lower() for w in PERSONAL.read_text(encoding="utf-8").splitlines() if w.strip()}

# The wordlist only has a handful of contractions, so cover the common ones
# explicitly. Possessives ('s, s') are handled by rule in is_known().
CONTRACTIONS = {
    "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't",
    "can't", "couldn't", "won't", "wouldn't", "shouldn't", "mustn't",
    "hasn't", "haven't", "hadn't", "ain't",
    "i'm", "i've", "i'll", "i'd",
    "you're", "you've", "you'll", "you'd",
    "he's", "he'll", "he'd", "she's", "she'll", "she'd", "it's", "it'll", "it'd",
    "we're", "we've", "we'll", "we'd", "they're", "they've", "they'll", "they'd",
    "that's", "that'll", "that'd", "there's", "there'll", "there'd", "here's",
    "what's", "what're", "what'll", "what'd", "who's", "who're", "who'll", "who'd",
    "where's", "when's", "why's", "how's", "let's", "o'clock", "y'all", "ma'am",
    "'tis", "'twas",
}

_DIGIT = re.compile(r"\d")


def normalize(word: str) -> str:
    """Curly apostrophes to straight, and drop any wrapping quotes/apostrophes."""
    return word.replace("’", "'").strip("'")


def is_known(word: str) -> bool:
    """Is this exact spelling (case-insensitive) in the dictionary?"""
    raw = word.replace("’", "'").lower()  # apostrophes intact, for the rules below
    if raw in CONTRACTIONS:
        return True
    if raw.endswith("'s") and raw[:-2] in WORDS:      # dog's
        return True
    if raw.endswith("s'") and raw[:-1] in WORDS:      # dogs'
        return True
    if raw.endswith("in'") and raw[:-1] + "g" in WORDS:  # goin'
        return True
    return raw.strip("'") in WORDS


def _at_sentence_start(context: str) -> bool:
    """True if the text before a word ends a sentence (or there is none)."""
    tail = context.rstrip()
    if not tail or "\n" in context[len(tail):]:  # start of document or of a line
        return True
    tail = tail.rstrip("\"'’”)")         # closing quotes after the full stop
    return not tail or tail[-1] in ".!?:"


def is_misspelled(word: str, context: str = "") -> bool:
    """
    Should this word be flagged? Conservative on purpose: anything that looks
    like a name, code, or acronym is left alone, because a wrong flag is more
    annoying than a missed one.

    `context` is the text before the word. It is only used to tell whether a
    capitalised word starts a sentence (check it) or sits mid-sentence
    (probably a proper noun, skip it). Pass "" if you don't have it.
    """
    w = normalize(word)
    if len(w) < 2:
        return False
    if _DIGIT.search(w) or "_" in w:
        return False  # versions, codes, identifiers
    if w.isupper():
        return False  # acronyms: NASA, ML, TODO
    if is_known(word):  # the raw word: is_known needs the apostrophes
        return False
    if w[0].isupper() and w[1:].islower() and not _at_sentence_start(context):
        return False  # capitalised mid-sentence: almost certainly a name
    return True


def frequency(word: str) -> int:
    """Corpus count for a word, 0 if unknown. Useful for ranking candidates."""
    return FREQUENCIES.get(normalize(word).lower(), 0)


def add_word(word: str) -> None:
    """Remember a word as correct, now and in future runs."""
    w = normalize(word).lower()
    if w in WORDS:
        return
    WORDS.add(w)
    DATA.mkdir(exist_ok=True)
    with open(PERSONAL, "a", encoding="utf-8") as f:
        f.write(w + "\n")
