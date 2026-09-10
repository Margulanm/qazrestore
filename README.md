# qazrestore

Restore the Kazakh-only letters that get lost when Kazakh is typed on a Russian keyboard — using a tiny character-level language model, not a lookup table.

## The problem

Kazakh Cyrillic has nine letters that don't exist on a standard Russian keyboard: **Ә Ғ Қ Ң Ө Ұ Ү Һ І**. Most keyboards, phones, and older systems in Kazakhstan only offer the 33 Russian letters, so people substitute the nearest one:

```
ә→а   ғ→г   қ→к   ң→н   ө→о   ұ,ү→у   һ→х   і→и
```

The result is readable to a person but is *not* the correctly spelled word — `қала` ("city") and `кала` are different strings. This quietly breaks Kazakh-language search, spell-checkers, screen readers, indexing, and any NLP pipeline downstream, and it's a routine fact of life in Kazakh social media, messaging, and legacy documents.

A dictionary can't fix this, because each substituted letter is genuinely ambiguous: a "у" in collapsed text might have been у, ұ, or ү, and the correct answer depends on the word — often on Kazakh vowel harmony, which is precisely the kind of pattern a statistical language model learns and a static lookup table cannot.

## The fix

`qazrestore` trains a character-level n-gram language model on a small Kazakh corpus and uses it to restore collapsed text with a Viterbi beam search: for every ambiguous letter it scores each possible original spelling against the model, picking the sequence that is most probable *as a whole* — not just the best guess at each position in isolation, which is what lets it use both left and right context.

```
$ python3 qazrestore.py restore "Kazakstan Respublikasynyn memlekettik tili — kazak tili"
# (Latin transliteration is out of scope — see Scope below)

$ python3 qazrestore.py restore "Казакстан Республикасынын мемлекеттик тили казак тили болып табылады."
Қазақстан Республикасының мемлекеттік тілі қазақ тілі болып табылады.
```

No downloads, no GPU, no dependencies beyond the Python standard library. Training + restoring + evaluating all run in well under a second.

## Why a language model, and does it actually help?

`qazrestore.py eval` trains on 80% of the bundled corpus, holds out the rest, collapses the held-out sentences to simulate real typing, restores them, and reports accuracy against two baselines:

| method                                       | char accuracy | sentence exact-match |
|-----------------------------------------------|:---:|:---:|
| identity (never restore anything)             | 86.5% | 0.0% |
| most-frequent letter, no context               | 91.5% | 16.1% |
| **n-gram language model + Viterbi (this repo)** | **96.8%** | **45.2%** |

The context-free "most frequent" baseline shows that *some* of this is easy (қ is simply more common than к in Kazakh text, for instance). The language model's further jump — especially in exact whole-sentence restoration, which nearly triples — is the part that requires actually looking at the surrounding letters, i.e. the part a lookup table cannot do. Run it yourself:

```
python3 qazrestore.py eval
```

## Usage

```
python3 qazrestore.py restore "<collapsed Kazakh text>"     # restore special letters
python3 qazrestore.py restore --collapse "<real Kazakh>"    # simulate the keyboard problem
python3 qazrestore.py eval                                  # accuracy vs. baselines
python3 test_qazrestore.py                                  # correctness tests
```

Bring your own corpus by replacing `corpus.txt` (one sentence per line, `#` for comments) — the model retrains from it automatically. The bundled corpus is a small hand-written set of ~150 Kazakh proverbs and everyday sentences, kept in the repo specifically so the project runs standalone with no external downloads; swapping in a larger corpus (e.g. Kazakh Wikipedia) is a one-line change and will only improve accuracy.

## Scope and honest limitations

- This restores **Cyrillic** Kazakh typed with Russian-keyboard substitutions. It does not do Latin-to-Cyrillic transliteration, spelling correction of genuine typos, or any language other than Kazakh.
- Recently borrowed/international words (e.g. `республика`) sometimes get over-corrected toward native vowel-harmony patterns (`респұблиқа`) because the bundled demo corpus is small and mostly native vocabulary. A larger training corpus with more loanwords fixes this directly — it's a data problem, not an architectural one.
- The model is a from-scratch character n-gram, not a pretrained LLM — deliberately, so the whole system is transparent, auditable, dependency-free, and retrains instantly on new text.

## Files

```
qazrestore.py         the model, restoration search, CLI, and evaluation
corpus.txt            training corpus (~150 Kazakh sentences)
test_qazrestore.py     correctness tests
```
