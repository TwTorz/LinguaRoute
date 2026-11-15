# LinguaRoute: full method and results

This page has the details behind the one-page write-up in the [README](../README.md). All numbers come from running `python3 app/train_language_model.py`, whose full output is saved in [evaluation_output.txt](evaluation_output.txt).

## Data

We use the [Tatoeba sentence corpus](https://tatoeba.org/en/downloads), specifically its detailed sentence export, which includes sentence IDs and contributor names. When we checked it on November 14, 2026, it had 2,041,097 English, 443,266 Spanish, 248,989 Japanese and 89,315 Mandarin Chinese sentences.

**Target languages** (English `eng`, Spanish `spa`, Japanese `jpn`, Mandarin `cmn`):
- We sampled only sentences with a recorded contributor name, so every sentence can be credited.
- We randomly sampled 5,000 sentences per language with a fixed seed (42), giving 20,000 in total.
- We removed exact duplicate texts, and any text that appeared under two different language labels, so the same sentence can't end up in both training and test data.
- We split the data into **12,000 train / 4,000 validation / 4,000 test**, stratified so each language has exactly 1,000 test sentences.

**Unknown-language data:**
- **Training "other" languages:** 1,800 sentences each from French, Portuguese, German and Italian.
- **Held out (never used in training):** 800 sentences each from Afrikaans and Dutch.
- We removed any of these texts that also appeared in the target-language data.

The corpus is downloaded to a local cache (`~/.cache/linguaroute/`) and is not stored in this repository. Two manifest files record every sentence used, with its ID, contributor and link: [tatoeba_attributions.csv](tatoeba_attributions.csv) for the target languages and [tatoeba_unknown_language_attributions.csv](tatoeba_unknown_language_attributions.csv) for the unknown-language samples.

## Models

**1. Language classifier.** TF-IDF features made from character chunks of 2–5 characters, fed into scikit-learn logistic regression. We compared two kinds of features on the **validation** split only:

| Features | Validation accuracy |
|---|---:|
| Word n-grams (1–2 words) | 73.50% |
| Character n-grams (2–5 characters) | **99.60%** |

Word features do badly because Japanese and Chinese don't put spaces between words, so a whole sentence looks like one unseen "word". After picking character n-grams, we retrained on train + validation and evaluated **once** on the test split.

**2. Writing-system check.** Before routing, the app narrows down which languages are possible using Unicode character names:

| Characters in the message | Possible languages |
|---|---|
| Latin letters only | English or Spanish |
| Japanese kana (with or without Chinese characters) | Japanese |
| Chinese characters only | Japanese or Mandarin |
| Mixed or other scripts | None; sent to human review |

**3. Supported vs. other detector.** This is a separate logistic regression on character n-grams. It is trained to tell the four supported languages apart from French, Portuguese, German and Italian, and it only runs on Latin-script text. A message scoring below 70% "supported" is flagged as possibly another language. The 70% threshold was chosen using validation data.

**4. Human-review rules.** The app suggests a queue only if a message passes all of these:
- at least 8 letters
- a supported, unmixed writing system
- a "supported" score of at least 70%
- language confidence of at least 85%

Everything else is held for a person.

Note: the demo app retrains its models on all 20,000 sampled sentences. That's fine for a demo, but the reported test scores come only from the evaluation script, where the test split is kept out of training.

## Results (held-out test split, 4,000 sentences)

| System | Correct | Accuracy |
|---|---:|---:|
| Baseline: always predict Mandarin | 1,000 / 4,000 | 25.00% |
| Character n-gram model (raw) | 3,974 / 4,000 | 99.35% |
| Model + writing-system check | 3,958 right, 3 wrong, 39 sent to review | — |
| Model + all review rules | 3,301 auto-routed, **0 wrong**; 699 sent to review | — |

The classes are balanced, so any always-the-same-answer baseline scores 25%.

**Per language (raw model):**

| Language | Precision | Recall | F1 | Test examples |
|---|---:|---:|---:|---:|
| English | 0.9960 | 0.9970 | 0.9965 | 1,000 |
| Japanese | 1.0000 | 0.9830 | 0.9914 | 1,000 |
| Mandarin Chinese | 0.9794 | 0.9970 | 0.9881 | 1,000 |
| Spanish | 0.9990 | 0.9970 | 0.9980 | 1,000 |

**Confusion matrix (raw model; rows = true language, columns = predicted):**

| | English | Spanish | Japanese | Mandarin |
|---|---:|---:|---:|---:|
| **English** | 997 | 0 | 0 | 3 |
| **Spanish** | 2 | 997 | 0 | 1 |
| **Japanese** | 0 | 0 | 983 | 17 |
| **Mandarin** | 2 | 1 | 0 | 997 |

## Unknown-language screening

| Check | Result (70% threshold) |
|---|---:|
| Validation: supported-language sentences accepted | 96.6% |
| Validation: French/Portuguese/German/Italian rejected | 99.6% |
| **Held-out Afrikaans rejected** (never seen in training) | **531 / 800 (66.4%)** |
| **Held-out Dutch rejected** (never seen in training) | **618 / 800 (77.2%)** |

The detector almost always catches the languages it was trained on, but only about two-thirds to three-quarters of languages it never saw. Afrikaans and Dutch share a lot of spelling patterns with English, which makes them hard. The score is not a calibrated probability.

## Error analysis

- **Japanese predicted as Mandarin (17 of the 26 raw errors).** These test sentences were very short (median 7 characters), and both languages use Chinese characters. 16 of the 17 contain Japanese kana, so the writing-system check corrects them; the one left uses only Chinese characters.
- **Short Latin text.** For the made-up message "Thanks!", the raw model guesses Mandarin, with under 50% confidence. A few characters don't carry enough pattern information. The writing-system check limits it to English/Spanish (it then picks English), and the 8-letter minimum sends it to a human.
- **The Japanese character 々 is not recognised.** This repetition mark (as in 人々, "people") isn't in our writing-system check's list, so it counts as an "unsupported writing system". That alone sent 13 Japanese test sentences to human review. It's safe, because nothing went to the wrong queue, but it's a bug.
- **Mixed scripts.** 20 Mandarin test sentences contain Latin letters (for example an English name inside a Chinese sentence), so they are sent to review instead of being routed.
- **Similar unknown languages pass.** About a third of Afrikaans and a quarter of Dutch sentences score as "supported".

## Limitations

- Tatoeba sentences are short, clean and carefully written. Real support messages contain typos, slang, product codes and mixed languages, so these scores are **not** a promise of real-world accuracy.
- Only four languages are supported, and unknown-language screening is far from perfect.
- The demo inbox messages are fictional examples written for the demo, not real customer data and not a benchmark.

## Licence and attribution

Tatoeba's [downloads page](https://tatoeba.org/en/downloads) states that sentence files are released under [CC BY 2.0 FR](https://creativecommons.org/licenses/by/2.0/fr/). See also Tatoeba's [terms of use](https://tatoeba.org/en/terms_of_use#section-6). We don't redistribute the corpus or sentence text; the two attribution manifests above credit every contributor whose sentence we used. Keep both manifests with any public release that uses these samples.
