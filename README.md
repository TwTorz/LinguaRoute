# LinguaRoute

**Track:** AI + Languages  
**Team:** Team Subaru: Tang Tang (Basis Independent McLean)

> LinguaRoute helps a **customer-support agent at a company with international customers** by taking **an incoming customer message** and giving back **its language (English, Spanish, Japanese or Mandarin), a confidence score and a suggested language queue, or a "needs a human" flag when it isn't sure.**

## The problem

When a support team serves customers in several languages, every message first has to reach someone who can read it. Sorting messages by hand is slow, and a message sent to the wrong queue waits longer for help. A confident wrong guess is worse than no guess, so the tool also has to know when *not* to decide.

## What we built

A Streamlit web app with three tabs:
- **Try a message:** paste text and get the language, a confidence score, and either a suggested queue or the reason it needs human review.
- **Demo inbox:** six fictional support messages processed together, with the model's answer shown next to the correct answer.
- **Results:** test scores, how the model works, and its limitations.

It is a prototype. It doesn't connect to a real ticket system or translate anything.

## What we added

We started from the suggested Beginner project, *language identification with Tatoeba*, and added:
- **Our own user and setting:** routing for a support team. Because a wrong route costs a customer time, we added **human-review rules**. A message is held for a person if it is too short (under 8 letters), the model is less than 85% confident, it mixes writing systems, or it may be a language we don't support.
- **Our own question:** *what happens with a language the model has never seen?* We trained a separate "supported vs. other language" detector using French, Portuguese, German and Italian. Then we tested it on **Afrikaans and Dutch, which it never saw during training**.

## How it works

**Data:** the [Tatoeba](https://tatoeba.org/en/downloads) sentence corpus. We used 5,000 sentences per language (20,000 in total), taking only sentences with a named contributor so each one can be credited. We removed duplicates, then split the data into 12,000 train, 4,000 validation and 4,000 test sentences (1,000 per language). The test set was used only once, at the end.

**Model:** TF-IDF on **character chunks** of 2–5 characters, plus logistic regression (scikit-learn). On the validation set, character chunks beat word-based features (99.60% vs 73.50%). Japanese and Chinese don't put spaces between words, and short patterns like "ción" or "th" are strong clues. A writing-system check then rules out impossible answers: Latin letters can only be English or Spanish, and Japanese kana can only be Japanese.

## Results

All results are on the 4,000 held-out test sentences.

| System | Result |
|---|---|
| Baseline: always guess "Mandarin" | 1,000 / 4,000 correct (**25.00%**) |
| Our model | 3,974 / 4,000 correct (**99.35%**); every language ≥ 98.3% recall |
| Our model + human-review rules | 3,301 auto-routed with **0 wrong**; 699 (17.5%) sent to a person |
| Unknown-language check (never-seen languages) | Caught **66.4% of Afrikaans** and **77.2% of Dutch** |

**Our weakest result** is the unknown-language check. About 1 in 3 Afrikaans and 1 in 4 Dutch sentences would slip through as "supported", because they share many spelling patterns with English.

**Where it fails:**
1. **Very short messages.** For "Thanks!", the raw model guessed *Mandarin* (under 50% confidence), because a few characters aren't enough evidence. The writing-system check changes the answer to English, and the 8-letter rule sends it to a person.
2. **Japanese read as Mandarin** (17 of the 26 errors). These were very short sentences (median 7 characters), and both languages use Chinese characters. 16 of the 17 contain Japanese kana, so the writing-system check fixes them.
3. **A bug we found:** the Japanese repetition mark 々 (as in 人々, "people") isn't recognised. It made 13 Japanese test sentences go to human review for no good reason. That's safe, but it's extra work for the agent.

**Honest caveat:** Tatoeba sentences are clean and well written. Real support messages have typos, slang and mixed languages, so 99% here is not a real-world promise.

Full method, per-language tables, confusion matrix and error analysis: [docs/DETAILS.md](docs/DETAILS.md).

## What we'd do next

1. Write a new set of realistic support messages (short, informal, with typos and mixed languages) and use it **only for testing**.
2. Improve unknown-language screening by training on more "other" languages, and fix the 々 bug.

## Run it

```bash
python3 -m pip install -r app/requirements.txt
python3 -m streamlit run app/app.py          # demo app
python3 app/train_language_model.py          # reproduce the test results
```

The first run downloads Tatoeba's export (about 305 MB) to `~/.cache/linguaroute/` and trains the models, which takes a few minutes. Our evaluation output is saved in [docs/evaluation_output.txt](docs/evaluation_output.txt).

## Sources

- **Dataset:** [Tatoeba sentence corpus](https://tatoeba.org/en/downloads), [CC BY 2.0 FR](https://creativecommons.org/licenses/by/2.0/fr/). Every sentence we used is credited to its contributor in [docs/tatoeba_attributions.csv](docs/tatoeba_attributions.csv) and [docs/tatoeba_unknown_language_attributions.csv](docs/tatoeba_unknown_language_attributions.csv). We don't redistribute the sentence text.
- **Libraries:** [scikit-learn](https://scikit-learn.org/), [pandas](https://pandas.pydata.org/), [Streamlit](https://streamlit.io/).
- **Pretrained models:** none. We trained our own models.
- **AI tools:** [Claude Code](https://claude.com/claude-code) (Anthropic) was used for planning, writing and debugging code, re-checking the evaluation results, and drafting this write-up. [GitHub Copilot](https://github.com/features/copilot) helped with code and documentation. We reviewed and tested all of the code and results.

*Built at the Veritas AI International High School Hackathon 2026 (November 14–15, 2026). Planning files (ideas, pitch outline, Git cheat sheet) were prepared before kickoff; all project code and model work was done during the hackathon.*
