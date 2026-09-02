"""
Minimal text editor with an autocorrect hook.

Run:  python3 editor.py

All ML lives in autocorrect.py. This file only does four things:
  1. shows a window with a plain Text widget
  2. notices when you finish typing a word (space / newline / punctuation)
  3. hands that word to autocorrect.correct() and, if it has a prediction,
     strikes the word through and shows the prediction beside it
  4. underlines every word spellcheck.is_misspelled() flags, re-checking
     the whole (small) document shortly after each change

A prediction is accepted with Tab or rejected with Esc while the cursor is
still right after it. Once you've typed on, left-click it to accept or
right-click to reject. Accepting swaps the prediction in for the word;
rejecting restores the word (still underlined, the dictionary doesn't know
it). Editing the struck-through word by hand drops the prediction.

Cmd+Z (Ctrl+Z on other platforms) undoes the last accept, then normal
typing. (Rejecting only removes display text, so there is nothing to undo.) Type :q on its own line and press Enter to quit, vim-style (:q! too).
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
FADED = "#cdb08f"  # struck-through original word
CREAM = "#f5cf9a"  # misspelled underline and the predicted word

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
        tk.Label(header, text="☕ Crema", font=header_font, fg=BROWN,
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
        # A word with a pending prediction: the original is struck through in
        # a faded brown, the prediction sits after it in cream. Each pending
        # pair also carries its own "orig<N>" / "sug<N>" tags so we can find
        # it again after the text around it has moved.
        self.text.tag_configure("struck", foreground=FADED, overstrike=True, overstrikefg=FADED)
        self.text.tag_configure("suggestion", foreground=CREAM)
        for tag in ("struck", "suggestion"):
            self.text.tag_bind(tag, "<Button-1>", self._on_click_accept)
            for seq in ("<Button-2>", "<Button-3>", "<Control-Button-1>"):
                self.text.tag_bind(tag, seq, self._on_click_reject)
        self._pending: dict[int, str] = {}  # id -> the original word
        self._next_id = 0
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

        # Widget-level bindings: run *before* the Text class handles the key.
        self.text.bind("<Return>", self._on_return)
        self.text.bind("<Tab>", self._on_tab)
        self.text.bind("<Escape>", self._on_escape)

        # Fires on any change to the buffer: typing, paste, undo, our replacements.
        self.text.bind("<<Modified>>", self._on_modified)
        # Moving the cursor off a half-typed word (arrows, click) should
        # check it too, even though the buffer didn't change.
        self.text.bind("<KeyRelease>", self._schedule_recheck)
        self.text.bind("<ButtonRelease-1>", self._schedule_recheck)

    # ---- word count and spell check -------------------------------------

    def _on_modified(self, event):
        if not self.text.edit_modified():
            return  # resetting the flag below re-fires this event; ignore that one
        n = len(self.text.get("1.0", "end-1c").split())
        for sid in self._pending:  # predictions aren't the user's words
            _, sug = self._ranges(sid)
            if sug:
                n -= len(self.text.get(*sug).split())
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
        self._drop_stale_predictions()
        self.text.tag_remove("misspelled", "1.0", "end")
        cursor = self.text.index("insert")
        lines = self.text.get("1.0", "end-1c").split("\n")
        for lineno, line in enumerate(lines, start=1):
            for m in ANY_WORD_RE.finditer(line):
                start, end = f"{lineno}.{m.start()}", f"{lineno}.{m.end()}"
                if end == cursor:
                    continue  # still being typed; judge it once it's finished
                names = self.text.tag_names(start)
                if "struck" in names or "suggestion" in names:
                    continue  # already has a prediction, or is one
                if is_misspelled(m.group(), line[:m.start()]):
                    self.text.tag_add("misspelled", start, end)

    # ---- keys -------------------------------------------------------------

    def _on_return(self, event):
        # Vim-style quit: a line that is just ":q" (or ":q!") followed by Enter.
        line = self.text.get("insert linestart", "insert lineend").strip()
        if line in (":q", ":q!"):
            self.root.destroy()
            return "break"  # swallow the keypress; nothing else runs

    def _on_tab(self, event):
        sid = self._prediction_at_cursor()
        if sid is None:
            return  # ordinary Tab
        self._accept(sid)
        return "break"

    def _on_escape(self, event):
        sid = self._prediction_at_cursor()
        if sid is None:
            return
        self._reject(sid)
        return "break"

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
        if "struck" in self.text.tag_names(start):
            return  # already has a prediction pending
        context = self.text.get("1.0", start)

        new = correct(word, context)
        if new is None or new == word:
            return
        self._show_prediction(start, end, word, new)

    # ---- predictions ------------------------------------------------------

    def _show_prediction(self, start: str, end: str, word: str, prediction: str):
        sid = self._next_id
        self._next_id += 1
        self._pending[sid] = word
        self.text.tag_add("struck", start, end)
        self.text.tag_add(f"orig{sid}", start, end)
        # The prediction is display-only until accepted, so keep it out of the
        # undo stack: Cmd+Z should undo what the user typed, not what we showed.
        self.text.configure(undo=False)
        self.text.insert(end, " " + prediction, ("suggestion", f"sug{sid}"))
        self.text.configure(undo=True)
        print(f"{word!r} -> {prediction!r}?")

    def _ranges(self, sid: int):
        """(original range, suggestion range) as index pairs, or None if gone."""
        o = self.text.tag_ranges(f"orig{sid}")
        s = self.text.tag_ranges(f"sug{sid}")
        return (tuple(map(str, o[:2])) if o else None,
                tuple(map(str, s[:2])) if s else None)

    def _prediction_at_cursor(self):
        """The pending prediction the cursor sits right after, if any."""
        here = {self.text.index("insert"), self.text.index("insert-1c")}
        for sid in self._pending:
            _, sug = self._ranges(sid)
            if sug and sug[1] in here:
                return sid
        return None

    def _prediction_at_event(self, event):
        for name in self.text.tag_names(f"@{event.x},{event.y}"):
            for prefix in ("orig", "sug"):
                if name.startswith(prefix) and name[len(prefix):].isdigit():
                    sid = int(name[len(prefix):])
                    if sid in self._pending:
                        return sid
        return None

    def _on_click_accept(self, event):
        sid = self._prediction_at_event(event)
        if sid is not None:
            self._accept(sid)
            return "break"

    def _on_click_reject(self, event):
        sid = self._prediction_at_event(event)
        if sid is not None:
            self._reject(sid)
            return "break"

    def _accept(self, sid: int):
        orig, sug = self._ranges(sid)
        if not orig or not sug:
            self._forget(sid)
            return
        prediction = self.text.get(*sug)[1:]  # drop the separating space
        self._remove_display_text(sug)
        # replace() is one undo step, so Cmd+Z brings the original word back.
        self.text.edit_separator()
        self.text.replace(*orig, prediction)
        self.text.edit_separator()
        print(f"{self._pending[sid]!r} -> {prediction!r}")
        self._forget(sid)

    def _reject(self, sid: int):
        _, sug = self._ranges(sid)
        if sug:
            self._remove_display_text(sug)
        self._forget(sid)

    def _remove_display_text(self, rng):
        # The prediction never entered the undo stack (see _show_prediction),
        # so it leaves the same way, otherwise undo would resurrect it as
        # plain text.
        self.text.configure(undo=False)
        self.text.delete(*rng)
        self.text.configure(undo=True)

    def _forget(self, sid: int):
        """Remove the pair's tags (leaves any text alone) and stop tracking it."""
        orig, _ = self._ranges(sid)
        if orig:
            self.text.tag_remove("struck", *orig)
        self.text.tag_delete(f"orig{sid}", f"sug{sid}")
        self._pending.pop(sid, None)
        self._schedule_recheck()

    def _drop_stale_predictions(self):
        # If the user edited the struck word, or deleted into the prediction,
        # the pair no longer makes sense: quietly take the prediction away.
        for sid, word in list(self._pending.items()):
            orig, sug = self._ranges(sid)
            if (not orig or not sug or self.text.get(*orig) != word
                    or not self.text.get(*sug).startswith(" ")):
                if sug:
                    self._remove_display_text(sug)
                self._forget(sid)


def main():
    root = tk.Tk()
    Editor(root)
    root.mainloop()


if __name__ == "__main__":
    main()
