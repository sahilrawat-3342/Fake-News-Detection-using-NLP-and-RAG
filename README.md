# TruthLens - AI-Powered Fake News Detection and Fact-Checking

TruthLens is a major college project built to detect fake news and verify suspicious claims. It combines a fast traditional machine learning model with an optional live RAG-based verification layer that searches the web and uses an LLM to reason over retrieved evidence.

The project can be used locally through a Streamlit web app. Users paste a news article or claim, run a quick fake-news prediction, and optionally perform deeper live verification with evidence and source links.

## Project Overview

TruthLens has two main verification layers:

1. **Layer 1: Fast ML Prediction**
   - Uses TF-IDF vectorization to convert article text into numerical features.
   - Uses Logistic Regression to classify content as Real or Fake.
   - Loads a saved trained model from `models/final_model.pkl`.
   - Gives a quick prediction with confidence score when available.

2. **Layer 2: Deep Live Verification**
   - Extracts the main factual claim from the input text.
   - Generates a focused web search query.
   - Retrieves live search results and article snippets.
   - Uses a Groq LLM to decide whether the claim is True, False, or Unverified.
   - Shows grounded reasoning and source links.

## Main Features

- Streamlit-based local web interface
- Fake news classification using TF-IDF and Logistic Regression
- Real/Fake prediction with confidence score
- Live RAG verification using web search and LLM reasoning
- Claim extraction from raw article text
- Source-backed verification results
- Model retraining from local CSV datasets
- Performance evaluation against baseline models
- Metrics export to JSON and CSV
- Visualization charts for model comparison

## Project Structure

```text
major_project-main/
+-- app.py                         # Main Streamlit web application
+-- train_model.py                 # Training script for the ML model
+-- evaluate_performance.py        # Script for model evaluation and metrics
+-- requirement.txt                # Python dependencies
+-- .env.example                   # Example environment variables
+-- data/
|   +-- Fake.csv                   # Fake news dataset
|   +-- True.csv                   # True news dataset
+-- models/
|   +-- final_model.pkl            # Saved trained model
+-- src/
|   +-- data_loader.py             # Text cleaning and dataset preparation
|   +-- model_engine.py            # Model wrapper for training and prediction
|   +-- rag_engine.py              # RAG, search, and LLM verification logic
|   +-- metrics_collector.py       # Metrics calculation and comparison
|   +-- metrics_visualizer.py      # Visualization generation
+-- visualizations/
|   +-- accuracy_f1_comparison.png
|   +-- metrics_heatmap.png
|   +-- model_comparison.png
|   +-- precision_recall.png
+-- performance_metrics.json       # Saved evaluation metrics
+-- performance_metrics.csv        # Metrics in CSV format
+-- METRICS_README.md              # Extra details about metrics system
```

## Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-learn
- NLTK
- Joblib
- LangChain
- Groq LLM
- SerpAPI
- Requests
- BeautifulSoup
- Python Dotenv

## Machine Learning Workflow

The Layer 1 model follows this pipeline:

1. Load fake and real news datasets from `data/Fake.csv` and `data/True.csv`.
2. Clean text by:
   - converting to lowercase
   - removing URLs
   - removing HTML tags
   - removing special characters
   - normalizing whitespace
3. Assign labels:
   - `0` = Fake
   - `1` = Real
4. Combine both datasets.
5. Remove duplicate text entries.
6. Split the data into training and testing sets.
7. Convert text into TF-IDF features.
8. Train a Logistic Regression classifier.
9. Save the trained pipeline to `models/final_model.pkl`.

## RAG Verification Workflow

The Layer 2 deep verification follows this process:

1. User enters a claim or news article.
2. The system extracts the most important verifiable claim.
3. A concise search query is generated.
4. Live search results are collected using SerpAPI.
5. Article snippets and page content are gathered.
6. A Groq LLM verifies the claim using only the retrieved evidence.
7. The app displays:
   - extracted claim
   - generated query
   - verdict
   - reasoning
   - source links

## Local Setup Instructions

### 1. Clone or Download the Project

Open the project folder in a terminal:

```powershell
cd path\to\major_project-main
```

### 2. Create a Virtual Environment

```powershell
python -m venv .venv
```

### 3. Activate the Virtual Environment

On Windows PowerShell:

```powershell
.\.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```powershell
pip install -r requirement.txt
```

### 5. Create Environment File

Copy the example environment file:

```powershell
copy .env.example .env
```

On macOS or Linux:

```bash
cp .env.example .env
```

### 6. Add API Keys for Deep Verification

Basic fake news prediction works without API keys because the trained model is already included.

For Deep Verify, open `.env` and add your keys:

```env
GROQ_API_KEY=your_groq_api_key
SERPAPI_KEY=your_serpapi_key
GOOGLE_API_KEY=your_google_api_key_optional
GOOGLE_CSE_ID=your_google_cse_id_optional
```

Required for Deep Verify:

- `GROQ_API_KEY`
- `SERPAPI_KEY`

Optional:

- `GOOGLE_API_KEY`
- `GOOGLE_CSE_ID`

## How to Run the Application

Start the Streamlit app:

```powershell
streamlit run app.py
```

After running the command, Streamlit will open the app in your browser. The local URL is usually:

```text
http://localhost:8501
```

## How to Use the App

1. Open the local Streamlit URL.
2. Paste a news article, claim, or suspicious content into the input box.
3. Click **EXECUTE PREDICTION** for fast ML classification.
4. The app will show whether the content is likely Real or Fake.
5. Click **DEEP VERIFY (LIVE RAG)** for evidence-backed verification.
6. The app will show a verdict, reasoning, and source links.

## Retraining the Model

The model can be retrained using the datasets inside the `data` folder.

### Option 1: Retrain from the App

Run the Streamlit app and click:

```text
INITIATE RETRAINING CYCLE
```

### Option 2: Retrain from Terminal

```powershell
python -c "from train_model import train_and_save_model; print(train_and_save_model())"
```

This will train the model again and save the updated model to:

```text
models/final_model.pkl
```

## Performance Evaluation

To run the full evaluation:

```powershell
python evaluate_performance.py
```

To generate visualizations:

```powershell
python evaluate_performance.py --visualize
```

This creates or updates:

- `performance_metrics.json`
- `performance_metrics.csv`
- `visualizations/model_comparison.png`
- `visualizations/accuracy_f1_comparison.png`
- `visualizations/precision_recall.png`
- `visualizations/metrics_heatmap.png`

## Evaluation Metrics

The project calculates:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Specificity
- Sensitivity
- Confusion matrix

It also compares the main TruthLens model with baseline models such as:

- Linear SVM
- Multinomial Naive Bayes
- Logistic Regression baseline

## Important Files

### `app.py`

Main Streamlit app. It handles the user interface, prediction button, retraining button, and deep verification output.

### `train_model.py`

Trains the TF-IDF + Logistic Regression pipeline using `Fake.csv` and `True.csv`.

### `src/data_loader.py`

Cleans raw text and prepares datasets for training.

### `src/rag_engine.py`

Handles the Deep Verify pipeline:

- claim extraction
- query generation
- web search
- source enrichment
- LLM-based verdict generation

### `src/metrics_collector.py`

Trains and evaluates models, calculates metrics, and stores results.

### `src/metrics_visualizer.py`

Generates charts from the saved model metrics.

## Example Use Case

A user sees a suspicious news article online. They copy the article text and paste it into TruthLens. First, they run the fast ML prediction to see whether the writing pattern looks real or fake. If they need more proof, they run Deep Verify. TruthLens then searches the web, extracts evidence, and gives a source-backed verdict.

## Notes

- The saved model is already available in `models/final_model.pkl`, so the app can run without retraining.
- Deep Verify requires internet access and valid API keys.
- The ML model is trained on the local datasets in the `data` folder.
- Results may vary depending on dataset quality, search results, and API availability.
- This project is intended for educational and research purposes.

## Project Summary

TruthLens is a complete fake news detection and fact-checking system. It demonstrates machine learning, natural language processing, web search, LLM-based reasoning, model evaluation, and an interactive user interface in one project.
