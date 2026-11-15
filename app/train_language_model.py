import csv
import io
import random
import tarfile
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

DATASET_URL = "https://downloads.tatoeba.org/exports/sentences_detailed.tar.bz2"
TARGET_LANGS = {
    "eng": "english",
    "spa": "spanish",
    "jpn": "japanese",
    "cmn": "mandarin",
}
UNKNOWN_TRAIN_LANGS = {"deu": "German", "fra": "French", "ita": "Italian", "por": "Portuguese"}
UNKNOWN_TEST_LANGS = {"afr": "Afrikaans", "nld": "Dutch"}
SAMPLES_PER_LANGUAGE = 5000
UNKNOWN_TRAIN_SAMPLES = 1800
UNKNOWN_TEST_SAMPLES = 800
RANDOM_SEED = 42
MIN_AUTO_ROUTE_CONFIDENCE = 0.85
SUPPORTED_LANGUAGE_THRESHOLD = 0.70
MIN_MESSAGE_LETTERS = 8
CACHE_DIR = Path.home() / ".cache" / "linguaroute"
SAMPLE_PATH = CACHE_DIR / "tatoeba_ccby_sample_with_authors.csv"
UNKNOWN_SAMPLE_PATH = CACHE_DIR / "tatoeba_unknown_language_sample.csv"
ATTRIBUTION_PATH = Path(__file__).resolve().parent.parent / "docs" / "tatoeba_attributions.csv"
UNKNOWN_ATTRIBUTION_PATH = Path(__file__).resolve().parent.parent / "docs" / "tatoeba_unknown_language_attributions.csv"


def _download_balanced_sample():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = CACHE_DIR / "sentences_detailed.tar.bz2"
    temporary_path = archive_path.with_suffix(".download")

    if not archive_path.exists():
        print("Downloading Tatoeba detailed sentence export (about 305 MB)...")
        with urllib.request.urlopen(DATASET_URL, timeout=180) as response:
            with temporary_path.open("wb") as destination:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    destination.write(chunk)
        temporary_path.replace(archive_path)

    rng = random.Random(RANDOM_SEED)
    seen = Counter()
    samples = {code: [] for code in TARGET_LANGS}
    with tarfile.open(archive_path, mode="r:bz2") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        if not members:
            raise RuntimeError("The Tatoeba archive did not contain a data file.")
        source = archive.extractfile(members[0])
        if source is None:
            raise RuntimeError("Could not read the Tatoeba sentence export.")
        text_stream = io.TextIOWrapper(source, encoding="utf-8")
        for line in text_stream:
            fields = line.rstrip("\n").split("\t", 5)
            if len(fields) < 4:
                continue
            sentence_id, language_code, text, author = fields[:4]
            if (
                language_code not in TARGET_LANGS
                or not text.strip()
                or not author.strip()
                or author == r"\N"
            ):
                continue

            seen[language_code] += 1
            row = {
                "id": sentence_id,
                "language_code": language_code,
                "text": text,
                "label": TARGET_LANGS[language_code],
                "author": author,
            }
            bucket = samples[language_code]
            if len(bucket) < SAMPLES_PER_LANGUAGE:
                bucket.append(row)
            else:
                position = rng.randrange(seen[language_code])
                if position < SAMPLES_PER_LANGUAGE:
                    bucket[position] = row

    insufficient = {
        TARGET_LANGS[code]: len(rows)
        for code, rows in samples.items()
        if len(rows) < SAMPLES_PER_LANGUAGE
    }
    if insufficient:
        raise RuntimeError(f"Not enough Tatoeba examples for the balanced sample: {insufficient}")

    sampled_rows = [row for code in TARGET_LANGS for row in samples[code]]
    sampled_df = pd.DataFrame(sampled_rows)
    unique_label_counts = sampled_df.groupby("text")["label"].nunique()
    unambiguous_texts = set(unique_label_counts[unique_label_counts == 1].index)
    sampled_df = sampled_df[sampled_df["text"].isin(unambiguous_texts)]
    sampled_df = sampled_df.drop_duplicates(subset="text", keep="first")
    per_language = sampled_df["label"].value_counts()
    balanced_count = int(per_language.min())
    sampled_df = pd.concat(
        [
            sampled_df[sampled_df["label"] == label].sample(
                n=balanced_count, random_state=RANDOM_SEED
            )
            for label in sorted(TARGET_LANGS.values())
        ],
        ignore_index=True,
    )
    sampled_df = sampled_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    sampled_rows = sampled_df.to_dict(orient="records")
    with SAMPLE_PATH.open("w", encoding="utf-8", newline="") as sample_file:
        writer = csv.DictWriter(sample_file, fieldnames=sampled_df.columns.tolist())
        writer.writeheader()
        writer.writerows(sampled_rows)

    ATTRIBUTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ATTRIBUTION_PATH.open("w", encoding="utf-8", newline="") as attribution_file:
        writer = csv.writer(attribution_file)
        writer.writerow(["sentence_id", "language", "contributor", "source_url"])
        for row in sampled_rows:
            sentence_url = f"https://tatoeba.org/en/sentences/show/{row['id']}"
            writer.writerow(
                [row["id"], row["label"], row["author"], sentence_url]
            )

    print("Tatoeba rows scanned:", dict(seen))
    print("Balanced sample rows:", len(sampled_rows))
    print("Attribution manifest:", ATTRIBUTION_PATH)


def load_labeled_data():
    if not SAMPLE_PATH.exists():
        _download_balanced_sample()
    df = pd.read_csv(SAMPLE_PATH, dtype={"id": str, "language_code": str, "author": str})
    print("Balanced Tatoeba sample:")
    print(df["label"].value_counts().sort_index())
    return df


def load_unknown_language_data():
    """Load negative examples and language-held-out examples for open-set rejection."""
    if not UNKNOWN_SAMPLE_PATH.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        archive_path = CACHE_DIR / "sentences_detailed.tar.bz2"
        if not archive_path.exists():
            _download_balanced_sample()

        limits = {
            **{code: UNKNOWN_TRAIN_SAMPLES for code in UNKNOWN_TRAIN_LANGS},
            **{code: UNKNOWN_TEST_SAMPLES for code in UNKNOWN_TEST_LANGS},
        }
        languages = {**UNKNOWN_TRAIN_LANGS, **UNKNOWN_TEST_LANGS}
        rng = random.Random(RANDOM_SEED + 1)
        seen = Counter()
        samples = {code: [] for code in limits}
        with tarfile.open(archive_path, mode="r:bz2") as archive:
            members = [member for member in archive.getmembers() if member.isfile()]
            if not members:
                raise RuntimeError("The Tatoeba archive did not contain a data file.")
            source = archive.extractfile(members[0])
            if source is None:
                raise RuntimeError("Could not read the Tatoeba sentence export.")
            text_stream = io.TextIOWrapper(source, encoding="utf-8")
            for line in text_stream:
                fields = line.rstrip("\n").split("\t", 5)
                if len(fields) < 4:
                    continue
                sentence_id, language_code, text, author = fields[:4]
                if (
                    language_code not in limits
                    or not text.strip()
                    or not author.strip()
                    or author == r"\N"
                ):
                    continue
                seen[language_code] += 1
                row = {
                    "id": sentence_id,
                    "language_code": language_code,
                    "text": text,
                    "language_name": languages[language_code],
                    "detector_split": "train_other"
                    if language_code in UNKNOWN_TRAIN_LANGS
                    else "heldout_language_test",
                    "author": author,
                }
                bucket = samples[language_code]
                if len(bucket) < limits[language_code]:
                    bucket.append(row)
                else:
                    position = rng.randrange(seen[language_code])
                    if position < limits[language_code]:
                        bucket[position] = row

        insufficient = {
            languages[code]: len(samples[code])
            for code in limits
            if len(samples[code]) < limits[code]
        }
        if insufficient:
            raise RuntimeError(
                f"Not enough attributed Tatoeba examples for the unknown-language test: {insufficient}"
            )

        data = pd.DataFrame(
            [row for code in limits for row in samples[code]]
        ).drop_duplicates(subset="text", keep="first")
        data.to_csv(UNKNOWN_SAMPLE_PATH, index=False)
        with UNKNOWN_ATTRIBUTION_PATH.open(
            "w", encoding="utf-8", newline=""
        ) as attribution_file:
            writer = csv.writer(attribution_file)
            writer.writerow(
                ["sentence_id", "language", "contributor", "source_url", "detector_split"]
            )
            for row in data.to_dict(orient="records"):
                writer.writerow(
                    [
                        row["id"],
                        row["language_name"],
                        row["author"],
                        f"https://tatoeba.org/en/sentences/show/{row['id']}",
                        row["detector_split"],
                    ]
                )
        print("Tatoeba out-of-scope sample rows:", len(data))
        print("Out-of-scope attribution manifest:", UNKNOWN_ATTRIBUTION_PATH)

    return pd.read_csv(
        UNKNOWN_SAMPLE_PATH,
        dtype={"id": str, "language_code": str, "author": str},
    )


def build_unknown_detector(known_df, unknown_train_df):
    """Train a separate known-vs-other classifier using other Latin languages."""
    known_rows = known_df[["text"]].copy()
    known_rows["is_supported"] = 1
    other_rows = unknown_train_df[["text"]].copy()
    other_rows["is_supported"] = 0
    training_rows = pd.concat([known_rows, other_rows], ignore_index=True)
    label_counts = training_rows.groupby("text")["is_supported"].nunique()
    unambiguous = set(label_counts[label_counts == 1].index)
    training_rows = training_rows[training_rows["text"].isin(unambiguous)]
    training_rows = training_rows.drop_duplicates(subset="text", keep="first")

    detector_vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(2, 5), min_df=2, sublinear_tf=True
    )
    features = detector_vectorizer.fit_transform(training_rows["text"])
    detector = LogisticRegression(C=1.0, max_iter=1000, solver="liblinear")
    detector.fit(features, training_rows["is_supported"])
    return detector_vectorizer, detector


def build_model(df=None, unknown_df=None):
    if df is None:
        df = load_labeled_data()
    if unknown_df is None:
        unknown_df = load_unknown_language_data()
    unknown_train_df = unknown_df[
        unknown_df["detector_split"] == "train_other"
    ]

    # Character chunks capture script clues such as Japanese kana and Chinese characters.
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2)
    X_vec = vectorizer.fit_transform(df["text"])

    model = LogisticRegression(max_iter=1000)
    model.fit(X_vec, df["label"])

    unknown_vectorizer, unknown_model = build_unknown_detector(
        df, unknown_train_df
    )
    return vectorizer, model, unknown_vectorizer, unknown_model


def predict_language(
    text,
    vectorizer,
    model,
    unknown_vectorizer=None,
    unknown_model=None,
):
    X_vec = vectorizer.transform([text])
    candidates, _ = script_candidates(text)
    if not candidates:
        return None, 0.0

    # The unknown-language detector is trained on other Latin-script languages,
    # so only apply it to Latin inputs, not Japanese or Mandarin.
    if (
        candidates == {"english", "spanish"}
        and unknown_vectorizer is not None
        and unknown_model is not None
    ):
        unknown_features = unknown_vectorizer.transform([text])
        supported_index = list(unknown_model.classes_).index(1)
        supported_score = float(
            unknown_model.predict_proba(unknown_features)[0][supported_index]
        )
        if supported_score < SUPPORTED_LANGUAGE_THRESHOLD:
            return None, supported_score

    probs = model.predict_proba(X_vec)[0]
    # Do not let a writing system unsupported by the input script win by model bias.
    filtered_probs = [
        probability if label in candidates else 0.0
        for label, probability in zip(model.classes_, probs)
    ]
    total = sum(filtered_probs)
    if total == 0:
        return None, 0.0
    filtered_probs = [probability / total for probability in filtered_probs]
    best_index = max(range(len(filtered_probs)), key=filtered_probs.__getitem__)
    pred = model.classes_[best_index]
    confidence = float(filtered_probs[best_index])
    return pred, confidence


def script_candidates(text):
    """Return plausible supported languages based on Unicode writing systems."""
    scripts = set()
    for character in text:
        if not character.isalpha():
            continue
        name = unicodedata.name(character, "")
        if "HIRAGANA" in name or "KATAKANA" in name:
            scripts.add("kana")
        elif "CJK UNIFIED IDEOGRAPH" in name or "CJK COMPATIBILITY IDEOGRAPH" in name:
            scripts.add("han")
        elif "LATIN" in name:
            scripts.add("latin")
        else:
            scripts.add("other")

    if not scripts:
        return None, "The message has too little recognizable writing to identify its language."
    if "other" in scripts:
        return None, "This message uses a writing system this prototype does not support."
    if scripts == {"latin"}:
        return {"english", "spanish"}, None
    if "kana" in scripts and scripts <= {"kana", "han"}:
        return {"japanese"}, None
    if scripts == {"han"}:
        return {"japanese", "mandarin"}, None
    return None, "This message mixes writing systems the prototype cannot safely distinguish."


def review_reason(text, confidence, possibly_unsupported=False):
    """Return a reason to ask a person to review, or None for a clear prediction."""
    candidates, script_reason = script_candidates(text)
    if script_reason:
        return script_reason
    if not candidates:
        return "The message does not match a supported writing system."
    letter_count = sum(character.isalpha() for character in text)
    if letter_count < MIN_MESSAGE_LETTERS:
        return "This message is very short or contains too little language text to route reliably."
    if possibly_unsupported:
        return "This may be a language outside the four supported languages; please verify it."
    if confidence < MIN_AUTO_ROUTE_CONFIDENCE:
        return "The model is not confident enough to route this message automatically."
    return None


def train_and_evaluate():
    df = load_labeled_data()
    unknown_df = load_unknown_language_data()
    known_texts = set(df["text"].astype(str))
    unknown_df = unknown_df[~unknown_df["text"].astype(str).isin(known_texts)]
    unknown_df = unknown_df.drop_duplicates(subset="text", keep="first")
    train_and_validation, test_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_SEED, stratify=df["label"]
    )
    train_df, validation_df = train_test_split(
        train_and_validation,
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=train_and_validation["label"],
    )
    negative_train = unknown_df[
        unknown_df["detector_split"] == "train_other"
    ]
    negative_heldout = unknown_df[
        unknown_df["detector_split"] == "heldout_language_test"
    ]
    negative_train_texts = set(negative_train["text"].astype(str))
    negative_heldout = negative_heldout[
        ~negative_heldout["text"].astype(str).isin(negative_train_texts)
    ]
    negative_fit, negative_validation = train_test_split(
        negative_train,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=negative_train["language_code"],
    )

    validation_detector_vectorizer, validation_detector = build_unknown_detector(
        train_df, negative_fit
    )
    supported_index = list(validation_detector.classes_).index(1)
    validation_known_scores = validation_detector.predict_proba(
        validation_detector_vectorizer.transform(validation_df["text"])
    )[:, supported_index]
    validation_other_scores = validation_detector.predict_proba(
        validation_detector_vectorizer.transform(negative_validation["text"])
    )[:, supported_index]
    known_acceptance = float(
        (validation_known_scores >= SUPPORTED_LANGUAGE_THRESHOLD).mean()
    )
    other_rejection = float(
        (validation_other_scores < SUPPORTED_LANGUAGE_THRESHOLD).mean()
    )
    print(
        "Supported-vs-other validation (threshold {:.0%}): {:.1%} of supported "
        "messages accepted; {:.1%} of training-negative languages rejected".format(
            SUPPORTED_LANGUAGE_THRESHOLD, known_acceptance, other_rejection
        )
    )

    # Compare feature types on validation data only; keep the held-out test unseen.
    candidates = {
        "word n-grams (1–2)": TfidfVectorizer(ngram_range=(1, 2), min_df=2),
        "character n-grams (2–5)": TfidfVectorizer(
            analyzer="char", ngram_range=(2, 5), min_df=2
        ),
    }
    for name, candidate_vectorizer in candidates.items():
        train_features = candidate_vectorizer.fit_transform(train_df["text"])
        candidate_model = LogisticRegression(max_iter=1000)
        candidate_model.fit(train_features, train_df["label"])
        validation_predictions = candidate_model.predict(
            candidate_vectorizer.transform(validation_df["text"])
        )
        validation_accuracy = accuracy_score(
            validation_df["label"], validation_predictions
        )
        print(f"Validation accuracy ({name}): {validation_accuracy:.4f}")

    # Character n-grams won validation, so refit on train + validation before test.
    training_data = pd.concat([train_df, validation_df], ignore_index=True)
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2)
    model = LogisticRegression(max_iter=1000)
    model.fit(vectorizer.fit_transform(training_data["text"]), training_data["label"])
    unknown_vectorizer, unknown_model = build_unknown_detector(
        training_data, negative_train
    )
    X_test = test_df["text"]
    y_test = test_df["label"]
    heldout_unknown = negative_heldout

    # The sample is balanced; this fixed-label baseline predicts Mandarin every time.
    baseline_pred = ["mandarin"] * len(y_test)
    baseline_acc = accuracy_score(y_test, baseline_pred)
    print("\nSingle-class baseline (always Mandarin) accuracy:", round(baseline_acc, 4))

    y_pred = model.predict(vectorizer.transform(X_test))
    acc = accuracy_score(y_test, y_pred)
    print("\nHeld-out Tatoeba test accuracy:", round(acc, 4))

    safeguarded_predictions = []
    auto_route = []
    for text in X_test:
        prediction, confidence = predict_language(
            text, vectorizer, model, unknown_vectorizer, unknown_model
        )
        safeguarded_predictions.append(prediction)
        auto_route.append(
            prediction is not None and review_reason(text, confidence) is None
        )
    auto_count = sum(auto_route)
    auto_errors = sum(
        predicted != actual
        for predicted, actual, accepted in zip(
            safeguarded_predictions, y_test, auto_route
        )
        if accepted
    )
    print(
        "Automatic-routing policy (supported-language score >= {:.0%}, language confidence >= {:.0%}, at least {} letters): "
        "{}/{} messages eligible; {} errors among eligible messages".format(
            SUPPORTED_LANGUAGE_THRESHOLD,
            MIN_AUTO_ROUTE_CONFIDENCE,
            MIN_MESSAGE_LETTERS,
            auto_count,
            len(test_df),
            auto_errors,
        )
    )

    unknown_scores = unknown_model.predict_proba(
        unknown_vectorizer.transform(heldout_unknown["text"])
    )[:, list(unknown_model.classes_).index(1)]
    unknown_score_series = pd.Series(unknown_scores, index=heldout_unknown.index)
    print("\nUnseen-language rejection (Afrikaans and Dutch were not detector training languages):")
    for language, indices in heldout_unknown.groupby("language_name").groups.items():
        language_scores = unknown_score_series.loc[list(indices)].to_numpy()
        rejected = int((language_scores < SUPPORTED_LANGUAGE_THRESHOLD).sum())
        print(
            f"  {language}: rejected {rejected}/{len(language_scores)} "
            f"({rejected / len(language_scores):.1%}) at {SUPPORTED_LANGUAGE_THRESHOLD:.0%} threshold"
        )

    print("\nPer-language report:")
    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred, labels=["english", "spanish", "japanese", "mandarin"])
    print("\nConfusion matrix:")
    print(cm)

    results = pd.DataFrame({"true": y_test, "pred": y_pred})
    mistakes = results[results["true"] != results["pred"]]
    print("\nMistake counts by true and predicted language:")
    print(mistakes.value_counts().to_string())


if __name__ == "__main__":
    train_and_evaluate()
