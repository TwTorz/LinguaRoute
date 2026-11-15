from train_language_model import load_labeled_data


def load_and_inspect():
    df = load_labeled_data()
    print("\nSample size:", len(df))
    print("Language counts:\n", df["label"].value_counts().sort_index())
    print("Unique sentence texts:", df["text"].nunique())
    print("Missing contributor names:", int(df["author"].isna().sum()))
    return df


if __name__ == "__main__":
    load_and_inspect()
