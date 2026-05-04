"""
Performance Metrics Collector Module
Collects, calculates, and compares performance metrics across different models.
Includes baseline comparisons: Logistic Regression, SVM, Naive Bayes
"""

import os
import json
import joblib
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from src.data_loader import DataProcessor


class MetricsCollector:
    """
    Comprehensive metrics collection and comparison system for fact-checking models.
    """

    def __init__(self, metrics_file: str = "performance_metrics.json"):
        """
        Initialize the metrics collector.
        
        Args:
            metrics_file: Path to store collected metrics
        """
        self.metrics_file = metrics_file
        self.metrics_data = self._load_metrics()
        self.processor = DataProcessor()

    def _load_metrics(self) -> Dict[str, Any]:
        """Load existing metrics or initialize new structure."""
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return self._initialize_metrics_structure()
        return self._initialize_metrics_structure()

    def _initialize_metrics_structure(self) -> Dict[str, Any]:
        """Initialize empty metrics structure."""
        return {
            "metadata": {
                "created": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "version": "1.0"
            },
            "models": {
                "truthlens_layer1": {},
                "svm_baseline": {},
                "naive_bayes_baseline": {},
                "logistic_regression_baseline": {}
            },
            "comparisons": [],
            "performance_history": []
        }

    def _save_metrics(self) -> None:
        """Save metrics to JSON file."""
        self.metrics_data["metadata"]["last_updated"] = datetime.now().isoformat()
        os.makedirs(os.path.dirname(self.metrics_file) if os.path.dirname(self.metrics_file) else ".", exist_ok=True)
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_data, f, indent=2)

    def prepare_data(self, fake_path: str = "data/Fake.csv", true_path: str = "data/True.csv") -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Prepare and split dataset for training/testing.
        
        Args:
            fake_path: Path to fake news dataset
            true_path: Path to true news dataset
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        fake_df = self.processor.prepare_dataset(fake_path)
        true_df = self.processor.prepare_dataset(true_path)

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

        return X_train, X_test, y_train, y_test

    def calculate_metrics(self, y_true: pd.Series, y_pred: pd.Series, y_pred_proba=None) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional, for ROC-AUC)
            
        Returns:
            Dictionary of calculated metrics
        """
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        }

        # Add ROC-AUC if probabilities available
        if y_pred_proba is not None:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred_proba))
            except Exception as e:
                print(f"Warning: Could not calculate ROC-AUC: {e}")
                metrics["roc_auc"] = None

        # Add confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics["confusion_matrix"] = {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        }

        # Add specificity and sensitivity
        metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        return metrics

    def train_truthlens_layer1(self, X_train: pd.Series, X_test: pd.Series, y_train: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
        """
        Train TruthLens Layer 1 model (TF-IDF + Logistic Regression).
        
        Args:
            X_train, X_test, y_train, y_test: Training and test sets
            
        Returns:
            Dictionary containing model and metrics
        """
        print("\n🔍 Training TruthLens Layer 1 (Logistic Regression + TF-IDF)...")
        
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
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

        metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
        
        result = {
            "model_name": "TruthLens Layer 1",
            "algorithm": "Logistic Regression + TF-IDF",
            "parameters": {
                "tfidf_ngram_range": (1, 2),
                "tfidf_max_df": 0.80,
                "tfidf_min_df": 3,
                "lr_solver": "liblinear",
                "lr_max_iter": 2000,
                "class_weight": "balanced"
            },
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }

        self.metrics_data["models"]["truthlens_layer1"] = result
        print(f"✅ TruthLens Layer 1 - Accuracy: {metrics['accuracy']:.4f}, F1-Score: {metrics['f1_score']:.4f}")
        
        return result

    def train_svm_baseline(self, X_train: pd.Series, X_test: pd.Series, y_train: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
        """
        Train SVM baseline model.
        
        Args:
            X_train, X_test, y_train, y_test: Training and test sets
            
        Returns:
            Dictionary containing model and metrics
        """
        print("\n🔍 Training SVM Baseline...")
        
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                max_df=0.80,
                min_df=3
            )),
            ("classifier", LinearSVC(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            ))
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        metrics = self.calculate_metrics(y_test, y_pred)
        
        result = {
            "model_name": "SVM Baseline",
            "algorithm": "Linear SVM + TF-IDF",
            "parameters": {
                "svm_kernel": "linear",
                "svm_max_iter": 2000,
                "class_weight": "balanced"
            },
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }

        self.metrics_data["models"]["svm_baseline"] = result
        print(f"✅ SVM Baseline - Accuracy: {metrics['accuracy']:.4f}, F1-Score: {metrics['f1_score']:.4f}")
        
        return result

    def train_naive_bayes_baseline(self, X_train: pd.Series, X_test: pd.Series, y_train: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
        """
        Train Naive Bayes baseline model.
        
        Args:
            X_train, X_test, y_train, y_test: Training and test sets
            
        Returns:
            Dictionary containing model and metrics
        """
        print("\n🔍 Training Naive Bayes Baseline...")
        
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                max_df=0.80,
                min_df=3
            )),
            ("classifier", MultinomialNB())
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

        metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
        
        result = {
            "model_name": "Naive Bayes Baseline",
            "algorithm": "Multinomial Naive Bayes + TF-IDF",
            "parameters": {
                "nb_alpha": 1.0
            },
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }

        self.metrics_data["models"]["naive_bayes_baseline"] = result
        print(f"✅ Naive Bayes Baseline - Accuracy: {metrics['accuracy']:.4f}, F1-Score: {metrics['f1_score']:.4f}")
        
        return result

    def generate_comparison_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive comparison report across all models.
        
        Returns:
            Dictionary containing comparison metrics
        """
        comparison = {
            "timestamp": datetime.now().isoformat(),
            "models_compared": [],
            "winner_by_metric": {},
            "detailed_comparison": []
        }

        models = self.metrics_data["models"]
        metrics_keys = ["accuracy", "precision", "recall", "f1_score"]

        # Extract model data
        for model_key, model_data in models.items():
            if model_data and "metrics" in model_data:
                comparison["models_compared"].append({
                    "key": model_key,
                    "name": model_data.get("model_name"),
                    "algorithm": model_data.get("algorithm")
                })
                comparison["detailed_comparison"].append({
                    "model_name": model_data.get("model_name"),
                    "metrics": model_data.get("metrics")
                })

        # Find winners for each metric
        for metric in metrics_keys:
            best_model = None
            best_score = -1
            for model_key, model_data in models.items():
                if model_data and "metrics" in model_data:
                    score = model_data["metrics"].get(metric, 0)
                    if score > best_score:
                        best_score = score
                        best_model = model_data.get("model_name")
            
            comparison["winner_by_metric"][metric] = {
                "model": best_model,
                "score": best_score
            }

        self.metrics_data["comparisons"].append(comparison)
        self._save_metrics()
        
        return comparison

    def print_summary_report(self) -> None:
        """Print a formatted summary report of all models."""
        print("\n" + "="*80)
        print("PERFORMANCE METRICS SUMMARY REPORT")
        print("="*80)

        models = self.metrics_data["models"]
        
        # Header
        print(f"\n{'Model Name':<30} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 80)

        # Model rows
        for model_key, model_data in models.items():
            if model_data and "metrics" in model_data:
                metrics = model_data["metrics"]
                print(f"{model_data.get('model_name', 'Unknown'):<30} "
                      f"{metrics.get('accuracy', 0):.4f}{'':>6} "
                      f"{metrics.get('precision', 0):.4f}{'':>6} "
                      f"{metrics.get('recall', 0):.4f}{'':>6} "
                      f"{metrics.get('f1_score', 0):.4f}{'':>6}")

        print("\n" + "="*80)
        print("COMPARISON - WINNERS BY METRIC")
        print("="*80)

        if self.metrics_data["comparisons"]:
            latest_comparison = self.metrics_data["comparisons"][-1]
            for metric, winner in latest_comparison["winner_by_metric"].items():
                print(f"{metric.upper():<20}: {winner['model']:<30} ({winner['score']:.4f})")

        print("\n" + "="*80)

    def export_metrics_json(self, output_path: str = "performance_metrics.json") -> None:
        """Export metrics to JSON file."""
        self._save_metrics()
        print(f"\n✅ Metrics saved to: {output_path}")

    def export_metrics_csv(self, output_path: str = "performance_metrics.csv") -> None:
        """Export metrics comparison to CSV file."""
        models = self.metrics_data["models"]
        rows = []

        for model_key, model_data in models.items():
            if model_data and "metrics" in model_data:
                row = {
                    "Model": model_data.get("model_name"),
                    "Algorithm": model_data.get("algorithm"),
                    "Accuracy": model_data["metrics"].get("accuracy"),
                    "Precision": model_data["metrics"].get("precision"),
                    "Recall": model_data["metrics"].get("recall"),
                    "F1-Score": model_data["metrics"].get("f1_score"),
                    "ROC-AUC": model_data["metrics"].get("roc_auc"),
                    "Specificity": model_data["metrics"].get("specificity"),
                    "Sensitivity": model_data["metrics"].get("sensitivity"),
                }
                rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"\n✅ Metrics CSV exported to: {output_path}")

    def run_full_evaluation(self, fake_path: str = "data/Fake.csv", true_path: str = "data/True.csv") -> Dict[str, Any]:
        """
        Run complete evaluation across all models.
        
        Args:
            fake_path: Path to fake news dataset
            true_path: Path to true news dataset
            
        Returns:
            Dictionary containing all evaluation results
        """
        print("\n" + "="*80)
        print("STARTING COMPREHENSIVE MODEL EVALUATION")
        print("="*80)

        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(fake_path, true_path)
        print(f"\n📊 Dataset prepared - Train size: {len(X_train)}, Test size: {len(X_test)}")

        # Train all models
        results = {
            "truthlens_layer1": self.train_truthlens_layer1(X_train, X_test, y_train, y_test),
            "svm_baseline": self.train_svm_baseline(X_train, X_test, y_train, y_test),
            "naive_bayes_baseline": self.train_naive_bayes_baseline(X_train, X_test, y_train, y_test),
        }

        # Generate comparison
        comparison = self.generate_comparison_report()

        # Print summary
        self.print_summary_report()

        return {
            "results": results,
            "comparison": comparison
        }
