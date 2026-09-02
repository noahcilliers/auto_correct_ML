# auto_correct

A bare-bones text editor whose only job is to feed the words you type into an
autocorrect function you write.

## Setup (once)

```bash
conda env create -f environment.yml
```

This makes a conda env called `auto_correct` with Python 3.12 and Tk. Add your ML packages (torch, etc.) to `environment.yml` and run `conda env update -f environment.yml` to pull them in.

## Run

```bash
conda activate auto_correct
python editor.py
```

The editor itself has no dependencies beyond `tkinter`, which the `tk` package in the env provides.

## Files

| File | What it is |
|---|---|
| `editor.py` | The window. A small header and a single plain `Text` widget, nothing else. Detects when a word is finished and calls the hook. Underlines misspelled words. Don't need to touch this. |
| `autocorrect.py` | **Your ML goes here.** One function: `correct(word, context) -> str | None`. |
| `spellcheck.py` | Dictionary lookup. `is_misspelled(word, context)` decides *whether* a word is wrong; it never fixes anything. Also `frequency(word)` for ranking candidates and `add_word(word)` for a personal dictionary. |
| `data/en_words.txt` | The wordlist: SymSpell's English frequency dictionary, ~83k words with corpus counts (MIT). |
| `data/personal_words.txt` | Optional, one word per line. Words you never want flagged. Created by `add_word()`. |

## Spell check

Every word in the document is checked against the dictionary shortly after
each edit, and misspelled ones get a cream underline. The word under the
cursor is left alone until you finish typing it.

The check is deliberately conservative. It skips one-letter words, anything
with a digit, ALL-CAPS acronyms, and Capitalised words mid-sentence (almost
always names). Possessives and common contractions pass. Real-word errors
("form" for "from") are not caught; that needs context and is a job for the
model later.

`correct()` uses the same check as a gate, so the model only ever sees words
the dictionary doesn't know.

## The hook

```python
def correct(word: str, context: str) -> str | None:
```

- Called once each time you finish a word (you type a space, newline, or punctuation right after it).
- `word` is the word you just typed. `context` is all document text before it.
- Return the replacement, or `None` to leave the word alone.
- The editor swaps the word in place. Cmd+Z undoes the correction.
- Quit vim-style: type `:q` (or `:q!`) on its own line and press Enter.
- Every correction is printed to the terminal as `'teh' -> 'the'` so you can watch what the model is doing.

`correct()` runs on the UI thread, so keep it fast (well under ~50 ms per word) or typing will stutter. Load your model once at module import time, not inside `correct()`.

## Why tkinter

It's the simplest possible choice: in the standard library, one widget, no build step, no browser, no packaging. If you later want inline underlines or suggestion popups, `Text` supports tags and you can add them without changing frameworks. Don't reach for PyQt, Kivy or a web stack for this; they add setup and give you nothing you need for the ML part.
