# Data plan

The model is a character seq2seq: misspelled word in, correct word out. No context.
The editor only sends it words that fail the dictionary check.

## Sources

| Source | What it is | Role |
|---|---|---|
| `data/en_words.txt` | SymSpell frequency list, ~83k correct words with counts. Already loaded. | Output vocabulary, sampling prior, raw material for synthetic corruption. |
| [Norvig `spell-errors.txt`](https://norvig.com/ngrams/spell-errors.txt) | ~7.8k correct words, ~40k real misspellings. Format: `four: forer, fours, fore*5` | Real error distribution. Validation set. |
| [Twitter Typo Corpus](https://luululu.com/tweet/typo-corpus-r1.txt) | 39k single-edit keyboard typos, tab-separated with context. | Optional, later. Adds mechanical typos to Norvig's cognitive ones. |

Not using the GitHub Typo Corpus: download is dead and it is sentence-level.

## Filtering (applied to everything)

- Lowercase. Keep words matching `^[a-z][a-z']*[a-z]$` with length 2-20, same rule as `model.py`.
- Drop pairs whose correct word is not in the dictionary. The model should only ever emit dictionary words.
- Drop pairs whose misspelling *is* in the dictionary. The editor gate never sends those to the model, so they are wasted or misleading.

## Splits

- Norvig: split by **correct word**, not by pair, roughly 80/20. Held-out words never appear in training in any form.
- Synthetic data needs no split. All validation is on held-out Norvig words. That measures the thing we care about: fixing words the model has never been trained to emit.

## Synthetic corruption

1. Sample a word from `en_words.txt` with weight `count ** 0.4` (already computed in `model.py`).
2. Apply 1-2 edits: insert, delete, substitute, transpose, double / undouble a letter.
3. Edit statistics come from the Norvig train split: align each pair with a Levenshtein backtrace, tally which operations and characters occur and where in the word, and sample edits from those tallies instead of uniformly. Synthetic errors then look like real ones but cover the whole vocabulary.
4. Leave ~10% of samples uncorrupted so the model learns to copy a word through unchanged.

## Training: two phases of pretraining, no separate finetune

1. **Synthetic only.** Train until word accuracy on held-out Norvig stops improving.
2. **Mixture.** Each batch is ~70% synthetic, ~30% Norvig train pairs. Lower learning rate. Early-stop on held-out Norvig accuracy.

Never train on Norvig alone. With only ~8k target words the decoder forgets the rest of the vocabulary and held-out accuracy drops.

## Inference guards

- Only return the model's output if it is in the dictionary. Otherwise return `None` and the word stays underlined.
- Use `spellcheck.frequency()` to break ties if we ever return more than one candidate.

## Metric

Word-level exact match on held-out Norvig words. Report it after every epoch in both phases.
