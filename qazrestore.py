#!/usr/bin/env python3
"""
qazrestore — restore Kazakh-specific letters that get lost when Kazakh is
typed on a Russian/ASCII keyboard.

The problem
-----------
Kazakh Cyrillic has nine letters that do not exist on a standard Russian
keyboard: Ә, Ғ, Қ, Ң, Ө, Ұ, Ү, Һ, І. Millions of Kazakh speakers write on
keyboards, phones, and legacy systems that only offer the 33 Russian
letters, so in practice they substitute the nearest Russian letter:

    ә→а   ғ→г   қ→к   ң→н   ө→о   ұ,ү→у   һ→х   і→и

The result ("collapsed" Kazakh) is readable to a human but is a different
string of characters than the correctly spelled word, which breaks search,
spell-checking, text-to-speech, indexing, and any downstream NLP that
expects real Kazakh orthography. A fixed dictionary lookup cannot repair
this because each collapsed letter is genuinely ambiguous (е.g. "у" could
have been у, ұ, or ү) and the correct choice depends on the surrounding
word — often on Kazakh's vowel-harmony pattern, which is exactly the kind
of thing a statistical language model captures and a lookup table cannot.

The fix
-------
This script trains a small character-level n-gram language model directly
from a Kazakh text corpus (correctly spelled, corpus.txt) and uses it to
restore collapsed text with a Viterbi search: at each ambiguous character
it scores every possible original letter by how likely it makes the
surrounding character sequence, according to the model, and keeps the best
sequence overall (not just the best choice at each position in isolation).

Everything is pure Python standard library — no downloads, no GPU, no
external dependencies. Train + restore + evaluate all run in under a
second on the bundled corpus.

Usage
-----
    python3 qazrestore.py restore "Kazak tili memlekettik til"
    python3 qazrestore.py restore --collapse "Қазақ тілі мемлекеттік тіл"
    python3 qazrestore.py eval
"""

import argparse
import math
import random
from collections import defaultdict

CORPUS_PATH = "corpus.txt"
ORDER = 4          # n-gram order (context length used to predict next char)
BEAM_WIDTH = 12     # Viterbi beam width
TEST_FRACTION = 0.2
RANDOM_SEED = 13

# Collapsed letter -> the original Kazakh letters it might stand for.
# A collapsed letter can always also just be itself (identity is listed
# first so ties favour the more common, non-special letter).
EXPANSIONS = {
    "а": ["а", "ә"],
    "г": ["г", "ғ"],
    "к": ["к", "қ"],
    "н": ["н", "ң"],
    "о": ["о", "ө"],
    "у": ["у", "ұ", "ү"],
    "х": ["х", "һ"],
    "и": ["и", "і"],
}

# Original Kazakh-specific letter -> the Russian-keyboard letter it collapses to.
COLLAPSE = {
    "ә": "а", "Ә": "А",
    "ғ": "г", "Ғ": "Г",
    "қ": "к", "Қ": "К",
    "ң": "н", "Ң": "Н",
    "ө": "о", "Ө": "О",
    "ұ": "у", "Ұ": "У",
    "ү": "у", "Ү": "У",
    "һ": "х", "Һ": "Х",
    "і": "и", "І": "И",
}


def collapse(text: str) -> str:
    """Simulate typing `text` on a keyboard that lacks the Kazakh-only letters."""
    return "".join(COLLAPSE.get(ch, ch) for ch in text)


def load_sentences(path: str):
    sentences = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            sentences.append(line)
    return sentences


DISCOUNT_K = 2.0


class CharNGramModel:
    """A character-level n-gram language model with recursive back-off
    interpolation (Jelinek-Mercer style, with a context-count-dependent
    mixing weight, similar in spirit to Witten-Bell smoothing).

    A single fixed high order with plain add-one smoothing collapses badly
    on a small corpus (most 4-character contexts are seen once or never),
    so instead we blend orders 1..`order`: an order is trusted in proportion
    to how much data was actually observed for that exact context, and
    otherwise the model falls back to shorter, better-populated contexts.
    """

    def __init__(self, order: int):
        self.order = order
        # counts[k][context][next_char] = count, for k = 1..order
        self.counts = {k: defaultdict(lambda: defaultdict(int)) for k in range(1, order + 1)}
        self.totals = {k: defaultdict(int) for k in range(1, order + 1)}
        self.vocab = set()

    def train(self, sentences):
        for s in sentences:
            padded = ("\x02" * (self.order - 1)) + s + "\x03"
            for ch in s:
                self.vocab.add(ch)
            for i in range(self.order - 1, len(padded)):
                nxt = padded[i]
                for k in range(1, self.order + 1):
                    context = padded[i - k:i]
                    self.counts[k][context][nxt] += 1
                    self.totals[k][context] += 1
        self.vocab.add("\x03")
        self.vocab_size = max(len(self.vocab), 1)

    def _prob(self, order: int, context: str, ch: str) -> float:
        if order == 0:
            return 1.0 / self.vocab_size
        ctx = context[-order:] if order else ""
        total = self.totals[order].get(ctx, 0)
        lower = self._prob(order - 1, context, ch)
        if total == 0:
            return lower
        counts = self.counts[order].get(ctx, {})
        p_high = counts.get(ch, 0) / total
        lam = total / (total + DISCOUNT_K)
        return lam * p_high + (1 - lam) * lower

    def logprob(self, context: str, ch: str) -> float:
        p = self._prob(self.order - 1, context, ch)
        return math.log(max(p, 1e-12))


def restore(text: str, model: CharNGramModel, beam_width: int = BEAM_WIDTH) -> str:
    """Restore the most likely original spelling of collapsed Kazakh `text`."""
    pad = "\x02" * (model.order - 1)
    # Each beam entry: (score, output_string, context_tail)
    beams = [(0.0, "", pad)]
    for ch in text:
        candidates = EXPANSIONS.get(ch.lower(), [ch])
        is_upper = ch.isupper()
        next_beams = []
        for score, out, ctx in beams:
            for cand in candidates:
                c = cand.upper() if is_upper else cand
                step_score = model.logprob(ctx, c)
                new_ctx = (ctx + c)[-(model.order - 1):]
                next_beams.append((score + step_score, out + c, new_ctx))
        next_beams.sort(key=lambda t: t[0], reverse=True)
        beams = next_beams[:beam_width]
    best = max(beams, key=lambda t: t[0])
    return best[1]


def naive_baseline(text: str) -> str:
    """Context-free baseline: always expand to the identity letter (never guess
    the special Kazakh letter). This is what you get with no language model at
    all, and it is the yardstick the n-gram model must beat."""
    return text


def most_frequent_baseline(text: str, unigram_pref):
    """Context-free baseline that expands each ambiguous letter to whichever
    candidate is globally most frequent in the training corpus, ignoring the
    surrounding context entirely."""
    out = []
    for ch in text:
        low = ch.lower()
        if low in EXPANSIONS:
            best = unigram_pref.get(low, low)
            out.append(best.upper() if ch.isupper() else best)
        else:
            out.append(ch)
    return "".join(out)


def build_unigram_preference(sentences):
    counts = defaultdict(lambda: defaultdict(int))
    for s in sentences:
        for ch in s:
            low = ch.lower()
            collapsed = collapse(low)
            if collapsed in EXPANSIONS:
                counts[collapsed][low] += 1
    pref = {}
    for collapsed, opts in counts.items():
        pref[collapsed] = max(opts, key=opts.get)
    return pref


def char_accuracy(pred: str, gold: str) -> float:
    if not gold:
        return 1.0
    n = min(len(pred), len(gold))
    correct = sum(1 for i in range(n) if pred[i] == gold[i])
    return correct / len(gold)


def run_eval():
    sentences = load_sentences(CORPUS_PATH)
    random.Random(RANDOM_SEED).shuffle(sentences)
    n_test = max(1, int(len(sentences) * TEST_FRACTION))
    test_sentences = sentences[:n_test]
    train_sentences = sentences[n_test:]

    model = CharNGramModel(ORDER)
    model.train(train_sentences)
    unigram_pref = build_unigram_preference(train_sentences)

    n = len(test_sentences)
    char_acc_model = char_acc_identity = char_acc_unigram = 0.0
    word_exact_model = word_exact_identity = word_exact_unigram = 0

    for gold in test_sentences:
        typed = collapse(gold)
        pred_model = restore(typed, model)
        pred_identity = naive_baseline(typed)
        pred_unigram = most_frequent_baseline(typed, unigram_pref)

        char_acc_model += char_accuracy(pred_model, gold)
        char_acc_identity += char_accuracy(pred_identity, gold)
        char_acc_unigram += char_accuracy(pred_unigram, gold)

        if pred_model == gold:
            word_exact_model += 1
        if pred_identity == gold:
            word_exact_identity += 1
        if pred_unigram == gold:
            word_exact_unigram += 1

    print(f"Test sentences: {n} (held out from {len(sentences)} total, "
          f"trained on {len(train_sentences)})\n")
    print(f"{'method':<28}{'char accuracy':<16}{'sentence exact-match'}")
    print(f"{'-'*28}{'-'*16}{'-'*20}")
    print(f"{'identity (no restoration)':<28}{char_acc_identity/n:<16.1%}"
          f"{word_exact_identity/n:.1%}")
    print(f"{'most-frequent (no context)':<28}{char_acc_unigram/n:<16.1%}"
          f"{word_exact_unigram/n:.1%}")
    print(f"{'n-gram LM + Viterbi (ours)':<28}{char_acc_model/n:<16.1%}"
          f"{word_exact_model/n:.1%}")

    print("\nSample restorations:")
    for gold in test_sentences[:5]:
        typed = collapse(gold)
        pred = restore(typed, model)
        mark = "OK " if pred == gold else "DIFF"
        print(f"  [{mark}] typed:    {typed}")
        print(f"          restored: {pred}")
        print(f"          gold:     {gold}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_restore = sub.add_parser("restore", help="restore Kazakh-only letters in collapsed text")
    p_restore.add_argument("text", help="text to restore (or to collapse, with --collapse)")
    p_restore.add_argument("--collapse", action="store_true",
                            help="instead of restoring, simulate typing this text on a "
                                 "Russian keyboard (show what a user would actually type)")
    p_restore.add_argument("--corpus", default=CORPUS_PATH)

    sub.add_parser("eval", help="evaluate restoration accuracy on a held-out split of corpus.txt")

    args = parser.parse_args()

    if args.command == "restore":
        if args.collapse:
            print(collapse(args.text))
            return
        sentences = load_sentences(args.corpus)
        model = CharNGramModel(ORDER)
        model.train(sentences)
        print(restore(args.text, model))
    elif args.command == "eval":
        run_eval()


if __name__ == "__main__":
    main()
