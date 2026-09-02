"""
Minimal text editor with an autocorrect hook.

Run:  python3 editor.py

All ML lives in autocorrect.py. This file only does four things:
  1. shows a window with a plain Text widget
  2. notices when you finish typing a word (space / newline / punctuation)
  3. hands that word to autocorrect.correct() and swaps in the result
  4. underlines every word spellcheck.is_misspelled() flags, re-checking
     the whole (small) document shortly after each change

Cmd+Z (Ctrl+Z on other platforms) undoes the last correction, then normal typing.
Type :q on its own line and press Enter to quit, vim-style (:q! works too).
"""
import re
import tkinter as tk
import tkinter.font as tkfont

from autocorrect import correct
from spellcheck import is_misspelled

# A word is letters, digits and apostrophes. Anything else typed right after
# a word (space, newline, punctuation) ends the word and triggers a check.
ANY_WORD_RE = re.compile(r"[\w'’]+")            # every word in a line
WORD_RE = re.compile(ANY_WORD_RE.pattern + "$")  # the word a line ends with

BROWN = "#a86f3b"  # header and body text colour
CREAM = "#f5cf9a"  # underline for misspelled words

RECHECK_DELAY_MS = 150  # wait for typing to pause before re-scanning


def is_word_boundary(event) -> bool:
    if event.keysym in ("space", "Return", "KP_Enter", "Tab"):
        return True
    ch = event.char
    if not ch:  # modifier, arrow and function keys have no char
        return False
    return ch.isprintable() and not ch.isalnum() and ch not in "'’"


class Editor:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Crema")
        root.geometry("800x600")

        font = ("Courier New", 14)
        # Header: coffee + name on the left, word count on the right.
        header = tk.Frame(root)
        header.pack(fill="x")
        header_font = tkfont.nametofont("TkDefaultFont").copy()
        header_font.configure(size=13, weight="bold")
        tk.Label(header, text="\u2615 Crema", font=header_font, fg=BROWN,
                 padx=12, pady=12).pack(side="left")
        count_font = tkfont.nametofont("TkDefaultFont").copy()
        count_font.configure(size=12)
        self.count = tk.Label(header, text="0 words", font=count_font, fg=BROWN, padx=12)
        self.count.pack(side="right")
        # "systemGridColor" is a subtle line in both light and dark mode.
        tk.Frame(root, height=1, bg="systemGridColor").pack(fill="x")

        # The text lives in a fixed-width box centred under the header, with a
        # 1px border. `width` is in characters of the font above.
        self.text = tk.Text(root, wrap="word", undo=True, font=font, width=64,
                            fg=BROWN, insertbackground=BROWN,  # text and cursor
                            padx=30, pady=24, highlightthickness=1,
                            highlightbackground="systemGridColor",
                            highlightcolor="systemGridColor")
        self.text.pack(expand=True, fill="y", pady=24)
        self.text.focus_set()

        # Misspelled words get a coloured underline (underlinefg needs Tk 8.6.6+).
        self.text.tag_configure("misspelled", underline=True, underlinefg=CREAM)
        self._recheck_job = None

        # Tk runs key bindings tag by tag: widget -> "Text" class -> toplevel -> all.
        # The "Text" class binding is the one that actually inserts the typed
        # character. We add our own tag right after it, so by the time our
        # handler runs the character is already in the buffer and the finished
        # word sits just before it. (Binding on the widget itself would run
        # *before* the character is inserted.)
        tags = list(self.text.bindtags())
        tags.insert(tags.index("Text") + 1, "AfterTextInsert")
        self.text.bindtags(tuple(tags))
        self.text.bind_class("AfterTextInsert", "<KeyPress>", self._on_key)

        # Widget-level binding: runs *before* the Text class inserts the newline.
        self.text.bind("<Return>", self._on_return)

        # Fires on any change to the buffer: typing, paste, undo, our replacements.
        self.text.bind("<<Modified>>", self._on_modified)
        # Moving the cursor off a half-typed word (arrows, click) should
        # check it too, even though the buffer didn't change.
        self.text.bind("<KeyRelease>", self._schedule_recheck)
        self.text.bind("<ButtonRelease-1>", self._schedule_recheck)

    def _on_modified(self, event):
        if not self.text.edit_modified():
            return  # resetting the flag below re-fires this event; ignore that one
        n = len(self.text.get("1.0", "end-1c").split())
        self.count.config(text=f"{n} word{'s' if n != 1 else ''}")
        self.text.edit_modified(False)  # re-arm so the next change fires again
        self._schedule_recheck()

    def _schedule_recheck(self, event=None):
        # Debounce: many keystrokes in quick succession -> one scan.
        if self._recheck_job is not None:
            self.root.after_cancel(self._recheck_job)
        self._recheck_job = self.root.after(RECHECK_DELAY_MS, self._recheck)

    def _recheck(self):
        self._recheck_job = None
        self.text.tag_remove("misspelled", "1.0", "end")
        cursor = self.text.index("insert")
        lines = self.text.get("1.0", "end-1c").split("\n")
        for lineno, line in enumerate(lines, start=1):
            for m in ANY_WORD_RE.finditer(line):
                end = f"{lineno}.{m.end()}"
                if end == cursor:
                    continue  # still being typed; judge it once it's finished
                if is_misspelled(m.group(), line[:m.start()]):
                    self.text.tag_add("misspelled", f"{lineno}.{m.start()}", end)

    def _on_return(self, event):
        # Vim-style quit: a line that is just ":q" (or ":q!") followed by Enter.
        line = self.text.get("insert linestart", "insert lineend").strip()
        if line in (":q", ":q!"):
            self.root.destroy()
            return "break"  # swallow the keypress; nothing else runs

    def _on_key(self, event):
        if not is_word_boundary(event):
            return

        # The boundary char is now the char just before the cursor. Look at the
        # text on its line up to that char and pull off the trailing word.
        boundary = self.text.index("insert-1c")
        line = boundary.split(".")[0]
        before = self.text.get(f"{line}.0", boundary)
        m = WORD_RE.search(before)
        if not m:
            return  # e.g. two spaces in a row, or space after punctuation

        word = m.group()
        start, end = f"{line}.{m.start()}", f"{line}.{m.end()}"
        context = self.text.get("1.0", start)

        new = correct(word, context)
        if new is None or new == word:
            return

        self.text.edit_separator()  # so one undo reverts just this correction
        self.text.replace(start, end, new)
        print(f"{word!r} -> {new!r}")


def main():
    root = tk.Tk()
    Editor(root)
    root.mainloop()


if __name__ == "__main__":
    main()
