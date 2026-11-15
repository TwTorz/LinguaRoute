# LinguaRoute

**Track:** AI + Languages  
**Team:** Team Subaru. Members: Tang Tang. School: Basis Independent McLean. 
**One-line summary:** LinguaRoute helps a global customer-support agent identify the language of an incoming message and choose a language-specific support queue.

## The problem

A support agent handling messages from international customers may need to identify a message's language before sending it to the right team. LinguaRoute explores whether a small text-classification model can help with that first routing step. It is a prototype, not a connection to a real support platform or a replacement for human review.

## What we built

The Streamlit demo has three parts: a single-message analyzer, a six-message support-inbox simulation, and a results/explanation page. It estimates English, Spanish, Japanese, or Mandarin, displays confidence, and recommends a queue only when the message passes the human-review checks. The polished inbox messages are fictional examples, not actual customer messages or training data. Two brief examples demonstrate when review is needed. We evaluated the model on Tatoeba sentences, not live support messages.

## What we added

We adapted a general language-identification task to a specific user and setting: a global support agent deciding which language queue should receive an incoming message. We also compared word-based and character-based text features and examined the model's errors instead of reporting only a single accuracy score.

## How it works

### Data

We use the [Tatoeba sentence corpus](https://tatoeba.org/en/downloads), specifically its detailed sentence export, which includes sentence IDs and contributor names. From the export we keep English (`eng`), Spanish (`spa`), Japanese (`jpn`), and Mandarin Chinese (`cmn`). The downloaded corpus contained 2,041,097 English, 443,266 Spanish, 248,989 Japanese, and 89,315 Mandarin sentences when checked on November 14, 2026. We only sampled rows with a recorded contributor name so we could provide attribution. For a balanced experiment, we sampled 5,000 sentences per language with a fixed random seed, removed exact duplicate text and text appearing under conflicting language labels, then split the 20,000 examples into 12,000 train (3,000 per language), 4,000 validation (1,000 per language), and 4,000 held-out test examples (1,000 per language). The source corpus is downloaded into the user's local cache; it is not included in this repository. The selected-sentence contributor/ID/link manifest is [docs/tatoeba_attributions.csv](docs/tatoeba_attributions.csv).

### Model

We convert text into character n-gram TF-IDF features (character sequences of length 2–5) and classify them with scikit-learn logistic regression. Character patterns can be useful language clues, including differences in writing systems. On our validation split, word n-grams scored 73.50% accuracy and character n-grams scored 99.60%. We selected character n-grams using validation data and evaluated the final language classifier on the held-out test split. Before suggesting a queue, the app narrows candidates using Unicode writing-system clues. A separate binary logistic-regression detector then screens Latin-script text as either “looks like one of the supported languages” or “possibly another language.” It was trained on English, Spanish, French, Portuguese, German, and Italian. Afrikaans and Dutch were kept out of this detector's training and reserved for an unseen-language check. Japanese kana and Han-only text use the script-aware four-language model; unsupported or mixed writing systems are sent for review.

### Demo and routing

The user interface is built with Streamlit. The inbox simulator processes several synthetic messages together and distinguishes suggested queues from messages held for a person. The results page explains the model, test results, and limitations. The app does not actually transfer tickets, translate messages, or contact a support team.

## Results

All numbers below are from our stratified held-out test split (4,000 examples), not from real support tickets. We selected features using the separate validation split and did not use the test split for tuning.

| System | Correct | Accuracy |
|---|---:|---:|
| Single-class baseline (always predicts Mandarin) | 1,000 / 4,000 | 25.00% |
| Character n-gram TF-IDF + logistic regression | 3,974 / 4,000 | 99.35% |

The four classes are balanced, so there is no more-common majority class; predicting any one fixed language gets 25% on this test set. Mandarin is the fixed label used for this simple baseline. Exact repeated text was removed before splitting to prevent the same sentence appearing in both train and test. The 99.35% score is the raw character model before the app's writing-system candidate filter and review gate.

Per-language results on our held-out test split:

| Language | Precision | Recall | F1 | Test examples |
|---|---:|---:|---:|---:|
| English | 0.9960 | 0.9970 | 0.9965 | 1,000 |
| Japanese | 1.0000 | 0.9830 | 0.9914 | 1,000 |
| Mandarin Chinese | 0.9794 | 0.9970 | 0.9881 | 1,000 |
| Spanish | 0.9990 | 0.9970 | 0.9980 | 1,000 |

The raw four-class classifier made 26 errors. The largest confusion was Japanese predicted as Mandarin (17 examples), because the raw model does not itself enforce script constraints. In a separate synthetic probe, it guessed Mandarin for the six-letter English message “Thanks!” with 47.8% confidence. After restricting Latin-script candidates to English/Spanish, the app estimates English instead, but still asks for review because six letters are insufficient for safe routing. The separate unknown-language detector rejected the user's Afrikaans example and the Portuguese probe, but it does not catch every unknown language.

### Unknown-language screening results

The detector's negative training examples were 1,800 sentences each from French, Portuguese, German, and Italian. The validation threshold was set to 70% “supported” score. On validation, it accepted 96.6% of target-language messages and rejected 99.6% of the negative examples from these same four non-target languages. To check generalization to languages it had never seen during training, we also tested 800 contributor-attributed Afrikaans and 800 Dutch sentences. It rejected 531/800 Afrikaans (66.4%) and 618/800 Dutch (77.2%) as potentially unsupported. So it caught the exact Afrikaans example, but still failed to reject a substantial share of these two unseen languages. These are Tatoeba sentence results, not customer-message results, and do not establish reliable open-set language identification.

### Human-review safeguard

The app does not recommend an automatic queue if a message has fewer than 8 letters, mixed or unsupported scripts, an unknown-language supported-score below 70%, or language confidence below 85%. On the held-out four-language test split, 3,301/4,000 messages passed all safeguards with 0 incorrect queue suggestions in this sample. Separately, the unknown-language detector rejected 66.4% of unseen Afrikaans and 77.2% of unseen Dutch sentences; the rest passed the unknown-language screen. The screening score is not a calibrated probability, and this result is not a guarantee of safe routing.

## What we'd do next

1. Build a small, clearly labeled set of new synthetic support messages—including very short messages, Japanese written mostly with Kanji, mixed-language text, and informal phrasing—and evaluate it separately without using it for training or tuning.
2. Expand and test unknown-language screening on additional languages, domains, and short support messages; the held-out Afrikaans/Dutch results show substantial room to improve.

## Run it locally

From the project directory, install the listed Python dependencies and start the app:

```bash
python3 -m pip install -r app/requirements.txt
python3 -m streamlit run app/app.py
```

To print the validation and held-out test metrics, run `python3 app/train_language_model.py`. The full output from our run (including the confusion matrix) is saved in [docs/evaluation_output.txt](docs/evaluation_output.txt).

The first prediction downloads and scans Tatoeba's detailed export (about 305 MB compressed), creates target and out-of-scope samples in `~/.cache/linguaroute/`, generates the attribution manifests in `docs/`, and trains both models. The archive and sampled sentence text are not stored in the repository. Keep both generated attribution manifests with any public release that uses these samples.

## Demo

- **Local demo:** Run the app using the instructions above.
- **Demo flow:** Try the single-message analyzer, review the fictional support-inbox examples to see clear routing suggestions and human-review cases, then show the results tab.

## Team

- **Team member(s):** See the team details at the top of this README.
- **GitHub:** [TwTorz/hackathon](https://github.com/TwTorz/hackathon)

## Sources and credits

- **Dataset:** [Tatoeba sentence corpus downloads](https://tatoeba.org/en/downloads), maintained by the Tatoeba community. The downloads page states that sentence files are released under [CC BY 2.0 FR](https://creativecommons.org/licenses/by/2.0/fr/). We use attributed samples from English, Spanish, Japanese, Mandarin, French, Portuguese, German, Italian, Afrikaans, and Dutch. [docs/tatoeba_attributions.csv](docs/tatoeba_attributions.csv) records the four target-language sample's sentence IDs/contributors; [docs/tatoeba_unknown_language_attributions.csv](docs/tatoeba_unknown_language_attributions.csv) records the detector's negative and language-held-out samples and their roles. We do not include dataset archives or sentence text in this repository. Read [Tatoeba's license and reuse terms](https://tatoeba.org/en/terms_of_use#section-6) and respect contributor attribution requirements.
- **Libraries:** [scikit-learn](https://scikit-learn.org/), [pandas](https://pandas.pydata.org/), and [Streamlit](https://streamlit.io/).
- **Pretrained models:** None. LinguaRoute trains its own TF-IDF + logistic regression model from the selected dataset splits.
- **AI tools:** Claude Code (Anthropic) was used for planning, code assistance, and debugging, as required by the project rules. GitHub Copilot assisted with code changes, evaluation, and documentation. The team reviewed and tested the resulting project.

## Built at

Veritas AI International High School Hackathon 2026 (November 14–15, 2026).

> **Project-work note:** The planning files (ideas, pitch outline, README template, and Git cheat sheet) were prepared before kickoff. Project code and model work were completed during the hackathon.
