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

Replace the body of correct() with your model. If your model is slow to
load, load it once at module level (here), not inside correct().
"""

from spellcheck import is_misspelled

# Placeholder so the editor visibly does something out of the box.
_TYPOS = {
    "teh": "the",
    "adn": "and",
    "recieve": "receive",
    "seperate": "separate",
    "definately": "definitely",
}


def correct(word: str, context: str) -> str | None:
    if not is_misspelled(word, context):
        return None  # spelt fine (or a name / acronym / code); leave it alone

    fixed = _TYPOS.get(word.lower())
    if fixed is None:
        return None
    if word[0].isupper():
        fixed = fixed.capitalize()
    return fixed
