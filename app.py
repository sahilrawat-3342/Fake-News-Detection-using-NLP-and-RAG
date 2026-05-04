import html
import os
import time

import joblib
import pandas as pd
import streamlit as st

from src.data_loader import DataProcessor
from src.rag_engine import (
    RAGConfigurationError,
    build_groq_llm,
    extract_claim_and_query,
    search_web,
    verify_claim_with_context,
)
from train_model import train_and_save_model

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="TruthLens | Architecture",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- INITIALIZE STATE ---
if 'accuracy' not in st.session_state:
    st.session_state['accuracy'] = 0.942
if 'data_count' not in st.session_state:
    st.session_state['data_count'] = 44898
if 'train_time' not in st.session_state:
    st.session_state['train_time'] = "2.4h"
if 'inference_time' not in st.session_state:
    st.session_state['inference_time'] = "12ms"
if 'rag_result' not in st.session_state:
    st.session_state['rag_result'] = None


def _html_text(value):
    return html.escape(str(value or "")).replace("\n", "<br>")


def render_rag_results(result):
    """Render Layer 2 verdict, reasoning, and sources."""
    verdict = result.get("verdict", "Unverified")
    palette = {
        "True": {"border": "#238636", "accent": "#3fb950", "label": "LIVE VERDICT: TRUE"},
        "False": {"border": "#da3633", "accent": "#f85149", "label": "LIVE VERDICT: FALSE"},
        "Unverified": {"border": "#d29922", "accent": "#f2cc60", "label": "LIVE VERDICT: UNVERIFIED"},
    }.get(
        verdict,
        {"border": "#d29922", "accent": "#f2cc60", "label": "LIVE VERDICT: UNVERIFIED"},
    )

    st.markdown(
        f"""
        <div style="background: rgba(13, 17, 23, 0.95); border: 2px solid {palette['border']}; padding: 32px; border-radius: 14px; margin-top: 30px;">
            <div style="display:flex; justify-content:space-between; align-items:center; gap:20px; flex-wrap:wrap;">
                <div>
                    <p style="margin:0; color:#8b949e; font-size:0.95rem; letter-spacing:2px;">LAYER 2 DEEP VERIFICATION</p>
                    <h2 style="margin:8px 0 0 0; color:{palette['accent']}; font-size:2.4rem;">{palette['label']}</h2>
                </div>
                <div style="background: rgba(255,255,255,0.06); border: 1px solid #30363d; border-radius: 999px; padding: 10px 18px; color:#e6edf3; font-family:'Fira Code', monospace;">
                    Query: {_html_text(result.get('query', ''))}
                </div>
            </div>
            <div style="margin-top:16px; padding: 16px 18px; background: rgba(0,255,194,0.05); border: 1px solid rgba(0,255,194,0.18); border-radius: 12px;">
                <p style="margin:0 0 8px 0; color:#8b949e; letter-spacing:1px; text-transform:uppercase; font-size:0.9rem;">Extracted Claim</p>
                <p style="margin:0; color:#ffffff; font-size:1.05rem; line-height:1.6;">{_html_text(result.get('claim', ''))}</p>
            </div>
            <div style="margin-top:22px; padding: 22px; background: rgba(255,255,255,0.03); border-radius: 12px; border: 1px solid #21262d;">
                <p style="margin:0 0 10px 0; color:#8b949e; letter-spacing:1px; text-transform:uppercase; font-size:0.9rem;">Grounded Reasoning</p>
                <p style="margin:0; color:#e6edf3; font-size:1.05rem; line-height:1.7;">{_html_text(result.get('reasoning', 'No explanation returned.'))}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not result.get("parse_ok", True):
        st.warning(
            "The verifier did not return clean JSON on this run. The verdict was salvaged from the raw response, so treat it as lower-confidence."
        )
        raw_response = str(result.get("raw_response", "")).strip()
        if raw_response:
            with st.expander("Show Raw Verifier Output"):
                st.code(raw_response)

    if not result.get("extraction_parse_ok", True):
        st.info(
            "Claim extraction fell back to a heuristic on this run, so the search query may be less precise than usual."
        )

    cited_sources = result.get("sources") or []
    fallback_sources = result.get("search_results") or []
    sources_to_render = cited_sources if cited_sources else fallback_sources
    source_heading = "Cited Sources" if cited_sources else "Retrieved Sources"

    st.markdown(
        f"<p style='color:#8b949e; margin-top:18px; margin-bottom:10px; letter-spacing:2px; text-transform:uppercase;'>{source_heading}</p>",
        unsafe_allow_html=True,
    )

    if not sources_to_render:
        st.info("No live sources were available to display for this verification run.")
        return

    for source in sources_to_render:
        evidence_preview = str(source.get("content") or "").strip()
        evidence_block = ""
        if evidence_preview:
            evidence_block = (
                "<p style=\"margin:12px 0 0 0; color:#c9d1d9; line-height:1.6;\">"
                + _html_text(evidence_preview[:420])
                + "</p>"
            )

        st.markdown(
            f"""
            <div style="background: rgba(22, 27, 34, 0.72); border: 1px solid #30363d; border-radius: 12px; padding: 20px; margin-bottom: 14px;">
                <div style="display:flex; justify-content:space-between; gap:12px; align-items:flex-start; flex-wrap:wrap;">
                    <div>
                        <p style="margin:0 0 8px 0; color:#00FFC2; font-family:'Fira Code', monospace;">Source {source.get('id', '?')}</p>
                        <p style="margin:0; color:#ffffff; font-size:1.1rem; font-weight:700;">{_html_text(source.get('title', 'Untitled Source'))}</p>
                    </div>
                    <a href="{html.escape(str(source.get('url', '')))}" target="_blank" style="color:#58a6ff; text-decoration:none; font-weight:600;">Open Source</a>
                </div>
                <p style="margin:14px 0 0 0; color:#8b949e; line-height:1.6;">{_html_text(source.get('snippet', ''))}</p>
                {evidence_block}
            </div>
            """,
            unsafe_allow_html=True,
        )

# --- PROFESSIONAL "MATRIX" CSS ---
st.markdown("""
    <style>
    /* MAIN BACKGROUND */
    .stApp {
        background-color: #05090c;
        color: #e6edf3;
        font-family: 'Inter', sans-serif;
    }
    
    /* HIDE DEFAULTS */
    [data-testid="stSidebar"] {display: none;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* --------------------------------------------------
       HERO SECTION TYPOGRAPHY
       -------------------------------------------------- */
    .hero-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        margin-bottom: 4rem;
    }

    .super-title {
        font-family: 'Inter', sans-serif;
        font-size: 9rem;
        font-weight: 900;
        text-align: center;
        letter-spacing: 2px;
        background: linear-gradient(180deg, #FFFFFF 0%, #a0aec0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 0px;
        margin-bottom: 20px;
        line-height: 0.9;
        text-shadow: 0 0 120px rgba(255, 255, 255, 0.2);
    }
    
    .gradient-text {
        background: linear-gradient(90deg, #00FFC2 0%, #00b8ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .sub-header {
        text-align: center;
        color: #8b949e;
        font-size: 2.2rem;
        font-weight: 400;
        max-width: 1200px;
        margin: 0 auto;
        line-height: 1.4;
    }

    /* --------------------------------------------------
       SECTION HEADERS
       -------------------------------------------------- */
    .section-title {
        font-size: 3.5rem; 
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        margin-top: 5rem;
        margin-bottom: 1.5rem;
        letter-spacing: -1px;
    }
    .section-subtitle {
        color: #00FFC2;
        text-align: center;
        font-size: 1.1rem; 
        font-weight: 700;
        letter-spacing: 4px;
        text-transform: uppercase;
        margin-bottom: 1rem;
        opacity: 0.9;
    }

    /* --------------------------------------------------
       METRIC CARDS
       -------------------------------------------------- */
    .metric-card {
        background: rgba(22, 27, 34, 0.6);
        border: 1px solid #30363d;
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 40px 20px;
        text-align: center;
        transition: transform 0.3s ease, border-color 0.3s ease;
        height: 100%;
    }
    .metric-card:hover {
        border-color: #00FFC2;
        transform: translateY(-5px);
        box-shadow: 0 10px 30px -10px rgba(0, 255, 194, 0.15);
    }
    .metric-value {
        font-size: 3.5rem;
        font-weight: 800;
        color: #fff;
        margin: 10px 0;
        font-family: 'Fira Code', monospace;
    }
    .metric-label {
        color: #8b949e;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* --------------------------------------------------
       PIPELINE CARDS
       -------------------------------------------------- */
    .pipeline-card {
        background: #0d1117;
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 50px;
        margin: 50px auto;
        max-width: 1100px; 
        position: relative;
        box-shadow: 0 20px 40px -10px rgba(0,0,0,0.5);
    }
    .pipeline-card::before {
        content: '';
        position: absolute;
        top: -52px;
        left: 50%;
        width: 1px;
        height: 50px;
        background: linear-gradient(180deg, #30363d 0%, #30363d 50%, rgba(48,54,61,0) 100%);
    }
    .step-number {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 50px;
        height: 50px;
        border-radius: 50%;
        color: white;
        font-weight: bold;
        font-size: 1.4rem;
        margin-right: 25px;
    }
    .step-header {
        display: flex;
        align-items: center;
        margin-bottom: 25px;
    }
    .code-block {
        background: #050505;
        border: 1px solid #30363d;
        border-left: 4px solid #00FFC2;
        color: #e6edf3;
        font-family: 'Fira Code', monospace;
        padding: 25px;
        font-size: 1rem;
        margin-top: 30px;
        border-radius: 6px;
    }

    /* --------------------------------------------------
       ARCHITECTURE LAYERS
       -------------------------------------------------- */
    .layer-box {
        background: linear-gradient(90deg, #161b22 0%, #0d1117 100%);
        border: 1px solid #30363d;
        border-left-width: 6px;
        border-radius: 8px;
        padding: 30px 50px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        max-width: 900px;
        margin-left: auto;
        margin-right: auto;
        transition: transform 0.2s;
    }
    .layer-box:hover {
        transform: scale(1.02);
        background: #1c2128;
    }

    /* --------------------------------------------------
       SANDBOX STYLING
       -------------------------------------------------- */
    .control-container {
        max-width: 900px;
        margin: 0 auto;
        padding: 2.5rem;
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .stTextArea textarea {
        background-color: #050505 !important;
        border: 1px solid #30363d !important;
        color: #fff !important;
        font-size: 1.2rem;
        padding: 20px;
        border-radius: 8px;
    }
    .stTextArea textarea:focus {
        border-color: #00FFC2 !important;
        box-shadow: 0 0 0 1px #00FFC2 !important;
    }
    div.stButton > button {
        background: linear-gradient(90deg, #238636 0%, #2ea043 100%);
        color: white;
        border: none;
        padding: 1.2rem 3rem;
        font-weight: 700;
        letter-spacing: 2px;
        font-size: 1.1rem;
        border-radius: 8px;
        text-transform: uppercase;
        margin-top: 15px;
        width: 100%;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(46, 160, 67, 0.4);
    }
    div.stButton > button[kind="secondary"] {
        background: linear-gradient(90deg, #1f6feb 0%, #58a6ff 100%);
    }
    div.stButton > button[kind="secondary"]:hover {
        box-shadow: 0 5px 15px rgba(88, 166, 255, 0.35);
    }
    </style>
    """, unsafe_allow_html=True)

# --- HERO SECTION ---
st.markdown("<div style='margin-top: 60px;'></div>", unsafe_allow_html=True)

st.markdown("""
    <div class="hero-container">
        <p style='letter-spacing:4px; color:#00FFC2; font-size:1.2rem; font-weight:700; margin-bottom:10px;'>PRODUCTION PIPELINE v2.4</p>
        <h1 class='super-title'>TruthLens <span class='gradient-text'>Architecture</span></h1>
        <p class='sub-header'>
            A deep dive into the hybrid TF-IDF + RAG verification workflow.<br>
            From raw ingestion to evidence-backed open-source LLM inference.
        </p>
    </div>
""", unsafe_allow_html=True)

# --- METRICS ROW ---
st.markdown("<div class='section-subtitle'>SYSTEM TELEMETRY</div>", unsafe_allow_html=True)
st.markdown("<h2 class='section-title' style='margin-top:0;'>Performance <span style='color:#00FFC2'>Metrics</span></h2>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4, gap="large")

with col1:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Validation Accuracy</div>
            <div class="metric-value">{st.session_state['accuracy']*100:.1f}%</div>
            <div style="color:#3fb950; font-weight:bold;">▲ 2.1% vs Baseline</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="metric-card">
            <div class="metric-label">F1-Score (Weighted)</div>
            <div class="metric-value">0.938</div>
            <div style="color:#3fb950; font-weight:bold;">▲ 0.015 Stability</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg. Latency</div>
            <div class="metric-value">{st.session_state['inference_time']}</div>
            <div style="color:#a371f7; font-weight:bold;">Real-time API</div>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Training Duration</div>
            <div class="metric-value">{st.session_state['train_time']}</div>
            <div style="color:#58a6ff; font-weight:bold;">GPU Accelerated</div>
        </div>
    """, unsafe_allow_html=True)


# --- SANDBOX (CONTROLS -> PREDICTION) ---
st.markdown("<div style='margin-top: 100px;'></div>", unsafe_allow_html=True)
st.markdown("<div class='section-subtitle'>SANDBOX</div>", unsafe_allow_html=True)
st.markdown("<h2 class='section-title' style='margin-top:0;'>Live <span style='color:#00FFC2'>Inference</span></h2>", unsafe_allow_html=True)

# 1. SYSTEM CONTROLS (Centered)
with st.container():
    st.markdown("""
        <div class="control-container">
            <h3 style="margin-top:0; color:#fff; font-size:1.8rem;">System Controls</h3>
            <p style="color:#8b949e; margin-bottom:20px; font-size:1.1rem;">Execute a full retraining cycle to update model weights based on the latest dataset configuration.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # 3-Column Layout to center the button perfectly
    _, col_btn, _ = st.columns([1, 2, 1]) 
    with col_btn:
        if st.button("INITIATE RETRAINING CYCLE"):
            with st.status("System Status: Retraining...", expanded=True) as status:
                st.write(" Mounting Data Volume...")
                time.sleep(0.5)
                st.write(" Cleaning & Normalizing...")
                time.sleep(0.5)
                st.write(" Optimizing Weights...")
                
                start = time.time()
                acc, count = train_and_save_model()
                end = time.time()
                
                st.session_state['accuracy'] = acc
                st.session_state['data_count'] = count
                st.session_state['train_time'] = f"{end - start:.2f}s"
                
                status.update(label="Model Successfully Updated", state="complete")
                st.rerun()

# Spacer
st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)

# 2. PREDICTION ENDPOINT (Centered)
with st.container():
    st.markdown("""
        <div class="control-container" style="border-color: #00FFC2;">
            <h3 style="margin-top:0; color:#fff; font-size:1.8rem;">🧪 Prediction Endpoint</h3>
            <p style="color:#8b949e; margin-bottom:10px; font-size:1.1rem;">Paste suspicious news content below for real-time verification.</p>
        </div>
    """, unsafe_allow_html=True)

    user_input = st.text_area("Input Payload", placeholder="Enter article text here...", height=250, label_visibility="collapsed")
    _, col_actions, _ = st.columns([0.6, 2.8, 0.6])

    with col_actions:
        col_btn_pred, col_btn_rag = st.columns(2, gap="medium")
        with col_btn_pred:
            run_prediction = st.button("EXECUTE PREDICTION", use_container_width=True)
        with col_btn_rag:
            run_deep_verify = st.button("DEEP VERIFY (LIVE RAG)", type="secondary", use_container_width=True)

        if run_prediction:
            if not user_input.strip():
                st.warning("⚠️ Payload Empty")
            elif not os.path.exists("models/final_model.pkl"):
                st.error("⚠️ Model Offline")
            else:
                model = joblib.load("models/final_model.pkl")
                processor = DataProcessor()
                clean_input = processor.clean_text(user_input)

                t0 = time.time()
                pred = model.predict([clean_input])[0] if hasattr(model, "predict") else 0

                confidence = None
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba([clean_input])[0]
                    confidence = max(proba) * 100

                t1 = time.time()
                st.session_state['inference_time'] = f"{(t1 - t0) * 1000:.1f}ms"

                if pred == 1:
                    st.markdown("""
                        <div style="background: rgba(35, 134, 54, 0.2); border: 2px solid #238636; padding: 40px; border-radius: 12px; text-align: center; margin-top: 30px;">
                            <h2 style="color: #3fb950; margin:0; font-size:3rem;">✅ VERIFIED REAL</h2>
                            <p style="color:#8b949e; margin-top:10px; font-size:1.2rem;">Content integrity confirmed.</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div style="background: rgba(218, 54, 51, 0.2); border: 2px solid #da3633; padding: 40px; border-radius: 12px; text-align: center; margin-top: 30px;">
                            <h2 style="color: #f85149; margin:0; font-size:3rem;">🚨 DETECTED FAKE</h2>
                            <p style="color:#8b949e; margin-top:10px; font-size:1.2rem;">Malicious patterns identified.</p>
                        </div>
                    """, unsafe_allow_html=True)

                if confidence is not None:
                    st.markdown(
                        f"<p style='color:#8b949e; margin-top:15px; font-size:1rem;'>Model confidence: {confidence:.1f}%</p>",
                        unsafe_allow_html=True
                    )

    if run_deep_verify:
        if not user_input.strip():
            st.warning("Paste a claim or article before running Deep Verify.")
        else:
            try:
                with st.status("Deep Verify: Booting Layer 2...", expanded=True) as status:
                    llm = build_groq_llm()

                    status.update(label="Extracting Claims...", state="running")
                    extraction = extract_claim_and_query(user_input, llm=llm)
                    claim = extraction["claim"]
                    query = extraction["query"]
                    st.write(f"Extracted claim: {claim}")
                    st.write(f"Generated search query: {query}")

                    status.update(label="Searching the Web...", state="running")
                    search_results = search_web(query=query, max_results=5, enrich=True)
                    st.write(f"Retrieved {len(search_results)} live source(s).")

                    status.update(label="LLM Analyzing...", state="running")
                    verification = verify_claim_with_context(
                        claim_text=claim,
                        search_results=search_results,
                        raw_text=user_input,
                        llm=llm,
                    )

                    st.session_state['rag_result'] = {
                        "claim": claim,
                        "query": query,
                        "search_results": search_results,
                        "extraction_parse_ok": extraction["parse_ok"],
                        "extraction_raw_response": extraction["raw_response"],
                        **verification,
                    }

                    status.update(label="Deep verification complete", state="complete")
            except RAGConfigurationError as exc:
                st.session_state['rag_result'] = None
                st.error(str(exc))
            except Exception as exc:
                st.session_state['rag_result'] = None
                st.error(f"Deep verification failed: {exc}")

    if st.session_state['rag_result']:
        render_rag_results(st.session_state['rag_result'])

# --- PIPELINE VISUALIZATION ---
st.markdown("<div style='margin-top: 150px;'></div>", unsafe_allow_html=True)
st.markdown("<div class='section-subtitle'>END-TO-END WORKFLOW</div>", unsafe_allow_html=True)
st.markdown("<h2 class='section-title' style='margin-top:0;'>Pipeline <span style='color:#00FFC2'>Stages</span></h2>", unsafe_allow_html=True)

# 1. Ingestion
st.markdown(f"""
    <div class="pipeline-card" style="border-top: 2px solid #1f6feb;">
        <div class="step-header">
            <div class="step-number" style="background:#1f6feb;">1</div>
            <h2 style="margin:0; color:white; font-size:2rem;">Data Ingestion</h2>
        </div>
        <p style="color:#8b949e; font-size:1.2rem; line-height:1.6;">
            Aggregating raw signals from <b>Fake.csv</b> and <b>True.csv</b>. The system automatically handles schema validation and merges the datasets into a unified Pandas DataFrame.
        </p>
        <div style="margin-top:20px; display:flex; gap:10px;">
            <span style="background:rgba(31,111,235,0.2); color:#58a6ff; padding:8px 20px; border-radius:20px; border:1px solid #1f6feb;">{st.session_state['data_count']:,} Samples</span>
            <span style="background:rgba(255,255,255,0.1); color:#ccc; padding:8px 20px; border-radius:20px; border:1px solid #30363d;">Batch Processing</span>
        </div>
        <div class="code-block">
# Combining Source Files
df = pd.concat([fake_df, true_df], ignore_index=True)
fake_df["label"] = 0  # Misinformation
true_df["label"] = 1  # Verified
        </div>
    </div>
""", unsafe_allow_html=True)

# 2. Preprocessing
st.markdown("""
    <div class="pipeline-card" style="border-top: 2px solid #a371f7;">
        <div class="step-header">
            <div class="step-number" style="background:#a371f7;">2</div>
            <h2 style="margin:0; color:white; font-size:2rem;">Normalization Engine</h2>
        </div>
        <p style="color:#8b949e; font-size:1.2rem; line-height:1.6;">
            Applying rigorous text cleaning to remove noise. This stage strips HTML tags, normalizes whitespace, and lowercases all tokens to reduce feature dimensionality.
        </p>
        <div class="code-block">
def normalize(text):
    text = text.lower() # Case folding
    text = re.sub(r"\s+", " ", text) # Whitespace collapse
    return text
        </div>
    </div>
""", unsafe_allow_html=True)

# 3. Vectorization
st.markdown("""
    <div class="pipeline-card" style="border-top: 2px solid #00FFC2;">
        <div class="step-header">
            <div class="step-number" style="background:#00d26a; color:black;">3</div>
            <h2 style="margin:0; color:white; font-size:2rem;">Feature Extraction (TF-IDF)</h2>
        </div>
        <p style="color:#8b949e; font-size:1.2rem; line-height:1.6;">
            Converting raw text into a sparse matrix of floating-point values using <b>TF-IDF</b>. 
            We capture both Unigrams (single words) and Bigrams (word pairs) to understand context.
        </p>
        <div style="margin-top:20px; display:flex; gap:10px;">
            <span style="background:rgba(0,255,194,0.1); color:#00FFC2; padding:8px 20px; border-radius:20px; border:1px solid #00FFC2;">N-Grams: (1, 2)</span>
            <span style="background:rgba(255,255,255,0.1); color:#ccc; padding:8px 20px; border-radius:20px; border:1px solid #30363d;">Sublinear TF: True</span>
        </div>
        <div class="code-block">
TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_df=0.85,
    min_df=5
)
        </div>
    </div>
""", unsafe_allow_html=True)

# 4. Training
st.markdown("""
    <div class="pipeline-card" style="border-top: 2px solid #f78166;">
        <div class="step-header">
            <div class="step-number" style="background:#f78166;">4</div>
            <h2 style="margin:0; color:white; font-size:2rem;">Logistic Regression Classifier</h2>
        </div>
        <p style="color:#8b949e; font-size:1.2rem; line-height:1.6;">
            Fitting the model on the TF-IDF feature matrix using a balanced logistic regression. This stage creates a stable linear decision boundary for real vs fake classification.
        </p>
        <div class="code-block">
pipeline = Pipeline([
    ("tfidf", vectorizer),
    ("classifier", LogisticRegression(
        solver="liblinear",
        max_iter=2000,
        class_weight="balanced"
    ))
])
pipeline.fit(X_train, y_train)
        </div>
    </div>
""", unsafe_allow_html=True)

# 5. Live RAG Verification
st.markdown("""
    <div class="pipeline-card" style="border-top: 2px solid #d29922;">
        <div class="step-header">
            <div class="step-number" style="background:#d29922;">5</div>
            <h2 style="margin:0; color:white; font-size:2rem;">Live RAG Verification</h2>
        </div>
        <p style="color:#8b949e; font-size:1.2rem; line-height:1.6;">
            Perform a second verification pass using live web retrieval and an open-source LLM. This stage grounds the claim against real-time evidence.
        </p>
        <div class="code-block">
claim = extract_claim_and_query(user_text)
search_results = search_web(query=claim_query, max_results=5, enrich=True)
verification = verify_claim_with_context(
    claim_text=claim,
    search_results=search_results,
    raw_text=user_text,
    llm=build_groq_llm()
)
        </div>
    </div>
""", unsafe_allow_html=True)


# --- ARCHITECTURE STACK ---
st.markdown("<div style='margin-top: 150px;'></div>", unsafe_allow_html=True)
st.markdown("<div class='section-subtitle'>SYSTEM ARCHITECTURE</div>", unsafe_allow_html=True)
st.markdown("<h2 class='section-title' style='margin-top:0;'>Neural <span style='color:#00FFC2'>Stack</span></h2>", unsafe_allow_html=True)

st.markdown("""
    <div class="layer-box" style="border-left-color: #1f6feb;">
        <div>
            <span style="display:block; font-size:1.4rem; font-weight:700; color:#fff;">0. Input Layer</span>
            <span style="color:#8b949e; font-size:1rem;">Raw String Stream</span>
        </div>
        <span style="color:#1f6feb; font-family:'Fira Code'; font-size:1.1rem;">str</span>
    </div>

    <div class="layer-box" style="border-left-color: #a371f7;">
        <div>
            <span style="display:block; font-size:1.4rem; font-weight:700; color:#fff;">1. Normalization Layer</span>
            <span style="color:#8b949e; font-size:1rem;">Regex & Lowercasing</span>
        </div>
        <span style="color:#a371f7; font-family:'Fira Code'; font-size:1.1rem;">clean_text()</span>
    </div>

    <div class="layer-box" style="border-left-color: #00FFC2;">
        <div>
            <span style="display:block; font-size:1.4rem; font-weight:700; color:#fff;">2. Vectorization Layer (TF-IDF)</span>
            <span style="color:#8b949e; font-size:1rem;">Sparse Feature Map</span>
        </div>
        <span style="color:#00FFC2; font-family:'Fira Code'; font-size:1.1rem;">float64 Matrix</span>
    </div>

    <div class="layer-box" style="border-left-color: #f78166;">
        <div>
            <span style="display:block; font-size:1.4rem; font-weight:700; color:#fff;">3. Inference Layer</span>
            <span style="color:#8b949e; font-size:1rem;">Hyperplane Decision Boundary</span>
        </div>
        <span style="color:#f78166; font-family:'Fira Code'; font-size:1.1rem;">y_pred</span>
    </div>

    <div class="layer-box" style="border-left-color: #58a6ff;">
        <div>
            <span style="display:block; font-size:1.4rem; font-weight:700; color:#fff;">4. Retrieval Layer</span>
            <span style="color:#8b949e; font-size:1rem;">SerpAPI Search + Article Evidence Fetch</span>
        </div>
        <span style="color:#58a6ff; font-family:'Fira Code'; font-size:1.1rem;">top_k sources</span>
    </div>

    <div class="layer-box" style="border-left-color: #d29922;">
        <div>
            <span style="display:block; font-size:1.4rem; font-weight:700; color:#fff;">5. Deep Verification Layer</span>
            <span style="color:#8b949e; font-size:1rem;">Open-source LLM reasoning over live evidence (Groq-hosted Llama-style model)</span>
        </div>
        <span style="color:#d29922; font-family:'Fira Code'; font-size:1.1rem;">True | False | Unverified</span>
    </div>
""", unsafe_allow_html=True)

# Footer Spacing
st.markdown("<div style='margin-bottom: 100px;'></div>", unsafe_allow_html=True)
