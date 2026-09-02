"""
The autocorrect hook. This is the only thing the editor imports.

The editor calls correct() every time you finish typing a word, i.e. the
moment you type a space, newline or punctuation mark right after it.

    word     the word you just typed, e.g. "teh"
    context  everything in the document before that word
             (handy for a language model; ignore it if you don't need it)

Return the corrected word, or None to leave it alone.

Replace the body of correct() with your model. If your model is slow to
load, load it once at module level (here), not inside correct().
"""

# Placeholder so the editor visibly does something out of the box.
_TYPOS = {
    "teh": "the",
    "adn": "and",
    "recieve": "receive",
    "seperate": "separate",
    "definately": "definitely",
}


def correct(word: str, context: str) -> str | None:
    fixed = _TYPOS.get(word.lower())
    if fixed is None:
        return None
    if word[0].isupper():
        fixed = fixed.capitalize()
    return fixed
