"""Minimal correctness tests for qazrestore. Run with: python3 test_qazrestore.py"""

import qazrestore as q


def test_collapse_matches_real_keyboard_substitution():
    assert q.collapse("Қазақ тілі") == "Казак тили"
    assert q.collapse("әғқңөұүһі") == "агкноуухи"
    assert q.collapse("ӘҒҚҢӨҰҮҺІ") == "АГКНОУУХИ"
    # Letters outside the special nine are left alone.
    assert q.collapse("Сәлем, әлем!") == "Салем, алем!"


def test_restore_recovers_known_sentences_from_the_corpus():
    sentences = q.load_sentences("corpus.txt")
    model = q.CharNGramModel(q.ORDER)
    model.train(sentences)

    # Sentences the model was trained on should restore losslessly: every
    # ambiguous letter has direct, unambiguous evidence in its own training
    # context, so there is no excuse for the model to get these wrong.
    checked = 0
    for s in sentences[:40]:
        typed = q.collapse(s)
        if typed == s:
            continue  # sentence had no ambiguous letters, not a useful check
        restored = q.restore(typed, model)
        assert restored == s, f"expected {s!r}, got {restored!r}"
        checked += 1
    assert checked > 10, "expected many ambiguous training sentences to check"


def test_restore_beats_identity_on_unseen_text():
    train_sentences = q.load_sentences("corpus.txt")[5:]
    model = q.CharNGramModel(q.ORDER)
    model.train(train_sentences)

    held_out = q.load_sentences("corpus.txt")[:5]
    lm_correct = sum(q.restore(q.collapse(s), model) == s for s in held_out)
    identity_correct = sum(q.collapse(s) == s for s in held_out)
    assert lm_correct >= identity_correct


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
