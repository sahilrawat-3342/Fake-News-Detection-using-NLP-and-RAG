# LaTeX Code Snippets for TruthLens Report

## Chapter 4: LAYER 1: Mathematical Formulation

### Model Training Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# TF-IDF + Logistic Regression Pipeline
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

# Train the model
pipeline.fit(X_train, y_train)
```

## Chapter 4: LAYER 2: The Verification Pipeline

### RAG Pipeline Core

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

def build_groq_llm(model_name="llama-3.3-70b-versatile"):
    """Initialize Groq LLM for RAG operations."""
    return ChatGroq(
        model_name=model_name,
        temperature=0.1,
        max_tokens=1024
    )

def extract_claim_and_query(text, llm):
    """Extract claim and generate search query."""
    prompt = ChatPromptTemplate.from_template("""
    Extract the main claim from this text and create a search query:

    Text: {text}

    Return JSON: {{"claim": "extracted claim", "query": "search query"}}
    """)

    response = llm.invoke(prompt.format(text=text))
    return json.loads(response.content)
```

## Chapter 5: IMPLEMENTATION DETAILS

### Data Preprocessing

```python
import pandas as pd
import re

class DataProcessor:
    def clean_text(self, text):
        """Lightweight TF-IDF-friendly text cleaning."""
        text = str(text).lower()
        # Remove URLs and HTML
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        text = re.sub(r'<.*?>', ' ', text)
        # Keep only alphanumeric
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
```

### Layer 1 Prediction (Backend)

```python
import joblib
import time

def predict_news_category(text, model_path="models/final_model.pkl"):
    """Layer 1 prediction with timing."""
    # Load model and processor
    model = joblib.load(model_path)
    processor = DataProcessor()

    # Preprocess input
    clean_input = processor.clean_text(text)

    # Make prediction with timing
    start_time = time.time()
    prediction = model.predict([clean_input])[0]

    # Get confidence if available
    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([clean_input])[0]
        confidence = max(proba) * 100

    inference_time = (time.time() - start_time) * 1000

    return {
        "prediction": prediction,  # 0=fake, 1=real
        "confidence": confidence,
        "inference_time_ms": inference_time
    }
```

### Streamlit UI (Frontend)

```python
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="TruthLens | Fact-Checking",
    page_icon="🛡️",
    layout="wide"
)

# Main interface
st.markdown("## 🛡️ TruthLens - Real-time Fact Checking")

# Input section
user_input = st.text_area(
    "Enter news article text:",
    placeholder="Paste suspicious news content here...",
    height=200
)

# Action buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("🔍 Quick Check (Layer 1)", use_container_width=True):
        # Layer 1 prediction logic
        pass

with col2:
    if st.button("🔬 Deep Verify (Layer 2)", use_container_width=True):
        # Layer 2 RAG verification logic
        pass
```

### Configuration and Security

```python
# .env configuration file
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_SEARCH_ENGINE_ID=your_custom_search_engine_id_here

# Streamlit secrets (secrets.toml)
[secrets]
GROQ_API_KEY = "your_groq_api_key_here"
GOOGLE_API_KEY = "your_google_api_key_here"
GOOGLE_SEARCH_ENGINE_ID = "your_custom_search_engine_id_here"

# Security best practices
def _read_secret(key: str) -> Optional[str]:
    """Securely read secrets from environment or Streamlit."""
    value = os.getenv(key)
    if value:
        return value

    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        return None

    return None
```

## LaTeX Integration Example

To include these in your LaTeX report, use:

```latex
\begin{lstlisting}[language=Python, caption=Data Preprocessing Pipeline, label=lst:data-preprocessing]
import pandas as pd
import re

class DataProcessor:
    def clean_text(self, text):
        """Lightweight TF-IDF-friendly text cleaning."""
        text = str(text).lower()
        # Remove URLs and HTML
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        text = re.sub(r'<.*?>', ' ', text)
        # Keep only alphanumeric
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
\end{lstlisting}
```

## Usage Instructions

1. **Copy the relevant snippet** for each section from above
2. **Format in LaTeX** using `\begin{lstlisting}[language=Python, caption=Your Caption, label=lst:your-label]`
3. **Place in the specified chapter and section** as indicated in your table
4. **Add cross-references** using `\ref{lst:your-label}` where needed

Would you like me to create a complete LaTeX file with all these snippets properly formatted and positioned?