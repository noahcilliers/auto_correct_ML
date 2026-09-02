"""
Minimal text editor with an autocorrect hook.

Run:  python3 editor.py

All ML lives in autocorrect.py. This file only does three things:
  1. shows a window with a plain Text widget
  2. notices when you finish typing a word (space / newline / punctuation)
  3. hands that word to autocorrect.correct() and swaps in the result

Cmd+Z (Ctrl+Z on other platforms) undoes the last correction, then normal typing.
Type :q on its own line and press Enter to quit, vim-style (:q! works too).
"""
import re
import tkinter as tk
import tkinter.font as tkfont

from autocorrect import correct

# A word is letters, digits and apostrophes. Anything else typed right after
# a word (space, newline, punctuation) ends the word and triggers a check.
WORD_RE = re.compile(r"[\w'’]+$")


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

        # Default font family, a few points bigger; padding keeps the text
        # away from the window edges so it reads as a column in the middle.
        font = tkfont.nametofont("TkFixedFont")
        font.configure(size=14)
        # Header: coffee + name, top left, coffee-brown.
        header_font = tkfont.nametofont("TkDefaultFont").copy()
        header_font.configure(size=13, weight="bold")
        tk.Label(root, text="\u2615 Crema", font=header_font, fg="#a86f3b",
                 anchor="w", padx=12, pady=12).pack(fill="x")
        # "systemGridColor" is a subtle line in both light and dark mode.
        tk.Frame(root, height=1, bg="systemGridColor").pack(fill="x")

        # The text lives in a fixed-width box centred under the header, with a
        # 1px border. `width` is in characters of the font above.
        self.text = tk.Text(root, wrap="word", undo=True, font=font, width=64,
                            padx=30, pady=24, highlightthickness=1,
                            highlightbackground="systemGridColor",
                            highlightcolor="systemGridColor")
        self.text.pack(expand=True, fill="y", pady=24)
        self.text.focus_set()

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
