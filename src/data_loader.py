import pandas as pd
import re

class DataProcessor:
    def __init__(self):
        pass

    def clean_text(self, text):
        """
        Lightweight, TF-IDF-friendly cleaning.
        Avoids linguistic leakage and train–test mismatch.
        """
        text = str(text).lower()

        # Remove URLs and HTML tags
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        text = re.sub(r'<.*?>', ' ', text)

        # Keep only letters and numbers so the vectorizer sees consistent tokens
        text = re.sub(r'[^a-z0-9\s]', ' ', text)

        # Collapse whitespace and strip leading/trailing spaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def prepare_dataset(self, path):
        df = pd.read_csv(path)

        if "text" not in df.columns:
            raise ValueError("Dataset must contain a 'text' column")

        df["cleaned_text"] = df["text"].apply(self.clean_text)
        return df
