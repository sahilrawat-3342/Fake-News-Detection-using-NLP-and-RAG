import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

class FakeNewsModel:
    def __init__(self):
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(
                stop_words='english',
                max_df=0.80,
                min_df=3,
                ngram_range=(1, 2),
                sublinear_tf=True
            )),
            ('classifier', LogisticRegression(
                solver='liblinear',
                max_iter=2000,
                class_weight='balanced',
                random_state=42
            ))
        ])

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)
        print("Model training complete.")

    def evaluate(self, x_test, y_test):
        predictions = self.model.predict(x_test)
        return classification_report(y_test, predictions)

    def save_model(self, filename='models/final_model.pkl'):
        joblib.dump(self.model, filename)

    def load_model(self, filename='models/final_model.pkl'):
        self.model = joblib.load(filename)

    def predict_single(self, text):
        return self.model.predict([text])[0]

    def predict_score(self, text):
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba([text])[0]
        return None