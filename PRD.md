# Product Requirements Document (PRD): TruthLens

## 1. Executive Summary

**Product Name:** TruthLens  
**Version:** 2.4 (Production Pipeline)  
**Product Type:** AI-Powered Fact-Checking Platform  
**Target Users:** Journalists, Fact-Checkers, Content Moderators, General Public  
**Release Date:** [Current Date]

TruthLens is a hybrid fact-checking system that combines traditional machine learning with cutting-edge Retrieval-Augmented Generation (RAG) technology to provide comprehensive news verification. The platform offers both rapid automated analysis and deep, evidence-backed verification for suspicious content.

## 2. Product Overview

### 2.1 Problem Statement
In an era of information overload and widespread misinformation, individuals and organizations need reliable tools to:
- Quickly verify the authenticity of news articles and claims
- Access evidence-based verification with transparent reasoning
- Distinguish between legitimate journalism and fabricated content
- Make informed decisions based on factual analysis

### 2.2 Solution
TruthLens provides a two-layer verification system:
- **Layer 1:** Fast ML-based classification using TF-IDF and Logistic Regression
- **Layer 2:** Deep verification using RAG with web search and LLM analysis

### 2.3 Value Proposition
- **Speed:** Sub-100ms inference for initial classification
- **Accuracy:** 94.2% validation accuracy with F1-score of 0.938
- **Transparency:** Evidence-backed reasoning with cited sources
- **User-Friendly:** Intuitive web interface with real-time results
- **Scalable:** Production-ready architecture supporting high-volume verification

## 3. Target Audience

### 3.1 Primary Users
- **Journalists and Media Professionals:** Need to verify sources and claims quickly
- **Content Moderators:** Platform teams requiring automated fact-checking
- **Fact-Checking Organizations:** NGOs and watchdog groups
- **Educators:** Teaching critical thinking and media literacy

### 3.2 Secondary Users
- **General Public:** Individuals wanting to verify news before sharing
- **Social Media Platforms:** Integration partners for content moderation
- **Research Institutions:** Academic researchers studying misinformation

## 4. Features and Capabilities

### 4.1 Core Features

#### 4.1.1 Real-Time Prediction (Layer 1)
- **Input:** Raw news article text
- **Processing:** TF-IDF vectorization + Logistic Regression
- **Output:** Binary classification (Real/Fake) with confidence score
- **Performance:** <100ms average latency
- **Accuracy:** 94.2% validation accuracy

#### 4.1.2 Deep Verification (Layer 2)
- **Input:** Suspicious claim or article
- **Processing:** 
  - Claim extraction using LLM
  - Web search for relevant sources
  - Evidence synthesis and reasoning
- **Output:** Verified verdict with:
  - Extracted claim
  - Grounded reasoning
  - Cited sources with snippets
  - Evidence previews

#### 4.1.3 Model Retraining
- **Trigger:** User-initiated or scheduled
- **Process:** Full pipeline retraining on latest dataset
- **Output:** Updated model weights and performance metrics

### 4.2 User Interface Features
- **Dark Theme:** Matrix-inspired professional design
- **Responsive Layout:** Optimized for desktop and mobile
- **Real-Time Metrics:** Live performance telemetry display
- **Interactive Sandbox:** Live testing environment
- **Status Indicators:** Clear feedback for all operations

### 4.3 Technical Features
- **Modular Architecture:** Separated concerns (data, model, RAG)
- **API Integration:** Google Custom Search, Groq LLM
- **Data Processing:** Robust text cleaning and normalization
- **Model Persistence:** Joblib-based model serialization
- **Error Handling:** Graceful degradation and user feedback

## 5. User Stories

### 5.1 Journalist Workflow
**As a journalist,** I want to verify a suspicious news article so that I can confidently report accurate information.

**Acceptance Criteria:**
- Input article text
- Receive instant classification (Real/Fake)
- Access detailed verification with sources
- Export verification results for publication

### 5.2 Content Moderator
**As a content moderator,** I want to check multiple articles quickly so that I can maintain platform integrity.

**Acceptance Criteria:**
- Batch processing capability
- High accuracy classification
- Minimal false positives/negatives
- Integration with existing moderation workflows

### 5.3 General User
**As a social media user,** I want to verify a viral post before sharing so that I don't spread misinformation.

**Acceptance Criteria:**
- Simple text input interface
- Clear, understandable results
- No technical expertise required
- Mobile-friendly interface

### 5.4 Fact-Checker
**As a professional fact-checker,** I want detailed evidence and reasoning so that I can build comprehensive verification reports.

**Acceptance Criteria:**
- Access to raw search results
- Transparent reasoning process
- Source credibility assessment
- Exportable verification reports

## 6. Technical Requirements

### 6.1 Functional Requirements

#### 6.1.1 Data Processing
- **REQ-001:** System shall clean and normalize input text
- **REQ-002:** System shall handle various text encodings and formats
- **REQ-003:** System shall remove URLs, HTML tags, and special characters
- **REQ-004:** System shall support English language processing

#### 6.1.2 Machine Learning Pipeline
- **REQ-005:** System shall use TF-IDF vectorization with n-gram range (1,2)
- **REQ-006:** System shall employ Logistic Regression with balanced class weights
- **REQ-007:** System shall achieve minimum 90% validation accuracy
- **REQ-008:** System shall provide prediction confidence scores

#### 6.1.3 RAG Engine
- **REQ-009:** System shall extract claims from input text using LLM
- **REQ-010:** System shall perform web search using SerpAPI
- **REQ-011:** System shall synthesize evidence from multiple sources
- **REQ-012:** System shall provide transparent reasoning for verdicts

#### 6.1.4 User Interface
- **REQ-013:** System shall provide web-based interface via Streamlit
- **REQ-014:** System shall display real-time performance metrics
- **REQ-015:** System shall support live model retraining
- **REQ-016:** System shall provide clear error messages and status updates

### 6.2 Non-Functional Requirements

#### 6.2.1 Performance
- **PERF-001:** Layer 1 inference < 100ms average
- **PERF-002:** Layer 2 verification < 30 seconds average
- **PERF-003:** Model training < 5 minutes for dataset < 50K samples
- **PERF-004:** System shall handle concurrent users without degradation

#### 6.2.2 Reliability
- **REL-001:** System availability > 99% during operational hours
- **REL-002:** Graceful handling of API failures (SerpAPI, Groq)
- **REL-003:** Automatic fallback for failed verifications
- **REL-004:** Data persistence for model artifacts

#### 6.2.3 Security
- **SEC-001:** No storage of user input data
- **SEC-002:** Secure API key management via environment variables
- **SEC-003:** Input sanitization to prevent injection attacks
- **SEC-004:** HTTPS-only communication for web interfaces

#### 6.2.4 Usability
- **USAB-001:** Interface shall be intuitive for non-technical users
- **USAB-002:** Clear visual feedback for all operations
- **USAB-003:** Responsive design for mobile and desktop
- **USAB-004:** Accessibility compliance (WCAG 2.1 AA)

## 7. System Architecture

### 7.1 High-Level Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │───▶│  Application    │───▶│   Data Layer    │
│                 │    │   Logic         │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ML Pipeline   │    │   RAG Engine    │    │  External APIs  │
│ (TF-IDF + LR)   │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 7.2 Component Details

#### 7.2.1 Data Layer (`src/data_loader.py`)
- **DataProcessor Class:** Text cleaning and dataset preparation
- **Input:** CSV files with text columns
- **Output:** Cleaned, normalized text data

#### 7.2.2 ML Pipeline (`train_model.py`)
- **TF-IDF Vectorizer:** Feature extraction with n-gram support
- **Logistic Regression:** Binary classification with class balancing
- **Pipeline:** Scikit-learn pipeline for consistent preprocessing

#### 7.2.3 RAG Engine (`src/rag_engine.py`)
- **Claim Extraction:** LLM-based claim identification
- **Web Search:** Google Custom Search integration
- **Evidence Synthesis:** Multi-source information aggregation
- **Reasoning:** LLM-powered verdict generation

#### 7.2.4 User Interface (`app.py`)
- **Streamlit App:** Web-based interactive interface
- **Real-time Updates:** Live metrics and status displays
- **Sandbox Environment:** Testing interface for both verification layers

### 7.3 Technology Stack
- **Frontend:** Streamlit, HTML/CSS
- **Backend:** Python 3.8+
- **ML Framework:** Scikit-learn, Joblib
- **LLM Integration:** LangChain, Groq API
- **Web Search:** SerpAPI
- **Data Processing:** Pandas, NLTK
- **Deployment:** Local/Cloud (Docker-ready)

## 8. Data Requirements

### 8.1 Training Data
- **Source:** Fake.csv and True.csv datasets
- **Format:** CSV with 'text' column
- **Size:** ~44,898 samples (balanced classes)
- **Preprocessing:** Text cleaning, deduplication, stratification

### 8.2 Input Data
- **Format:** Plain text (news articles, claims)
- **Encoding:** UTF-8
- **Size Limit:** 10,000 characters per input
- **Languages:** English (primary)

### 8.3 Output Data
- **Classification:** Binary (0=Fake, 1=Real) with confidence
- **Verification:** JSON with verdict, reasoning, sources
- **Metrics:** Accuracy, F1-score, latency measurements

## 9. Dependencies and Environment

### 9.1 Python Packages
```
streamlit>=1.28.0
pandas>=2.0.0
scikit-learn>=1.3.0
nltk>=3.8.0
joblib>=1.3.0
langchain>=0.1.0
langchain-groq>=0.1.0
python-dotenv>=1.0.0
requests>=2.31.0
beautifulsoup4>=4.12.0
serpapi>=0.1.4
```

### 9.2 External APIs
- **SerpAPI:** Web search functionality
- **Groq API:** LLM inference for claim extraction and reasoning
- **Environment Variables Required:**
  - `SERPAPI_KEY`
  - `GROQ_API_KEY`

### 9.3 System Requirements
- **Python:** 3.8 or higher
- **RAM:** 4GB minimum, 8GB recommended
- **Storage:** 1GB for models and data
- **Network:** Internet connection for API calls

## 10. Success Metrics

### 10.1 Performance Metrics
- **Accuracy:** >94% on validation set
- **F1-Score:** >0.93 (weighted average)
- **Latency:** <100ms for Layer 1, <30s for Layer 2
- **Uptime:** >99% availability

### 10.2 User Engagement Metrics
- **User Satisfaction:** >4.5/5 rating
- **Task Completion:** >95% successful verifications
- **Return Usage:** >70% repeat user rate
- **Feature Adoption:** >80% use both verification layers

### 10.3 Business Impact Metrics
- **Misinformation Reduction:** Measurable decrease in fake news sharing
- **Time Savings:** >50% reduction in manual fact-checking time
- **Cost Efficiency:** < $0.01 per verification
- **Scalability:** Support for 1000+ verifications per hour

## 11. Risk Assessment

### 11.1 Technical Risks
- **API Dependency:** Service outages from SerpAPI/Groq APIs
- **Model Drift:** Performance degradation over time
- **Data Quality:** Inconsistent or biased training data
- **Scalability:** Performance issues under high load

### 11.2 Business Risks
- **Regulatory Compliance:** Data privacy and content moderation laws
- **Ethical Concerns:** Potential misuse for censorship
- **Competition:** Emerging AI fact-checking solutions
- **User Trust:** Maintaining credibility and transparency

### 11.3 Mitigation Strategies
- **Fallback Systems:** Local processing when APIs unavailable
- **Continuous Monitoring:** Performance tracking and alerting
- **Regular Updates:** Model retraining with fresh data
- **Ethical Guidelines:** Clear usage policies and transparency

## 12. Future Roadmap

### 12.1 Phase 2 Features
- **Multi-language Support:** Extend beyond English
- **Batch Processing:** Verify multiple articles simultaneously
- **API Endpoints:** REST API for integration
- **Advanced Analytics:** Detailed reporting and insights

### 12.2 Phase 3 Features
- **Real-time Monitoring:** Social media integration
- **Collaborative Verification:** Multi-user fact-checking workflows
- **Source Credibility Scoring:** Authority assessment for sources
- **Mobile Application:** Native iOS/Android apps

### 12.3 Technical Improvements
- **Model Enhancements:** Transformer-based architectures
- **Edge Deployment:** On-device inference capabilities
- **Federated Learning:** Privacy-preserving model updates
- **Explainability:** Advanced interpretability features

## 13. Conclusion

TruthLens represents a significant advancement in automated fact-checking technology, combining the speed of traditional machine learning with the depth of modern AI systems. By providing both rapid classification and evidence-backed verification, the platform addresses critical needs in today's information ecosystem.

The hybrid approach ensures high accuracy while maintaining computational efficiency, making it suitable for both individual users and large-scale content moderation operations. With a focus on transparency, usability, and ethical deployment, TruthLens is positioned to become a trusted tool in the fight against misinformation.

---

**Document Version:** 1.0  
**Last Updated:** April 27, 2026  
**Author:** AI Assistant  
**Review Status:** Draft</content>
<parameter name="filePath">c:\Users\Sahil\Desktop\major_project-main\major_project-main\PRD.md