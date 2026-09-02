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
| `editor.py` | The window. A small header and a single plain `Text` widget, nothing else. Detects when a word is finished and calls the hook. Don't need to touch this. |
| `autocorrect.py` | **Your ML goes here.** One function: `correct(word, context) -> str | None`. |

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
