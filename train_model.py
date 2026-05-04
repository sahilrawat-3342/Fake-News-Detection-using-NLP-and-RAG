import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from src.data_loader import DataProcessor


def train_and_save_model(
    fake_path="data/Fake.csv",
    true_path="data/True.csv",
    model_path="models/final_model.pkl"
):
    os.makedirs("models", exist_ok=True)
    processor = DataProcessor()

    fake_df = processor.prepare_dataset(fake_path)
    true_df = processor.prepare_dataset(true_path)

    fake_df["label"] = 0
    true_df["label"] = 1

    fake_df = fake_df[["cleaned_text", "label"]].rename(columns={"cleaned_text": "text"})
    true_df = true_df[["cleaned_text", "label"]].rename(columns={"cleaned_text": "text"})

    df = pd.concat([fake_df, true_df], ignore_index=True)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_df=0.80,
            min_df=3,
            sublinear_tf=True
        )),
        ("classifier", LogisticRegression(
            solver="liblinear",
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    print("Validation report:\n", report)

    joblib.dump(pipeline, model_path)

    return acc, len(df)
