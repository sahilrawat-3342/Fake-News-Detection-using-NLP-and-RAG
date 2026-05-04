import json
import os
import re
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from src.data_loader import DataProcessor

load_dotenv(override=True)

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_MAX_SEARCH_RESULTS = 5
DEFAULT_FETCH_TIMEOUT = 6
MAX_PAGE_CHARS = 2200
MAX_CONTEXT_CHARS = 12000
GOOGLE_CUSTOM_SEARCH_URL = "https://www.googleapis.com/customsearch/v1"
SERPAPI_ENGINE_DEFAULT = "google"
SERPAPI_LOCATION_DEFAULT = "India"
SERPAPI_HL_DEFAULT = "hi"
SERPAPI_GL_DEFAULT = "in"
SERPAPI_GOOGLE_DOMAIN_DEFAULT = "google.co.in"
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


class RAGConfigurationError(RuntimeError):
    """Raised when Layer 2 configuration is incomplete."""


def _read_secret(key: str) -> Optional[str]:
    """Read a secret from environment variables or Streamlit secrets."""
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


def get_groq_api_key() -> str:
    """Return the Groq API key or raise a helpful configuration error."""
    api_key = _read_secret("GROQ_API_KEY")
    if api_key:
        return api_key

    raise RAGConfigurationError(
        "Missing GROQ_API_KEY. Add it to your .env file or Streamlit secrets."
    )


def get_google_api_key() -> str:
    """Return the Google Custom Search API key or raise a config error."""
    api_key = _read_secret("GOOGLE_API_KEY") or _read_secret("GOOGLE_CUSTOM_SEARCH_API_KEY")
    if api_key:
        return api_key.strip()

    raise RAGConfigurationError(
        "Missing GOOGLE_API_KEY. Add it to your .env file or Streamlit secrets."
    )


def get_google_cse_id() -> str:
    """Return the Google Programmable Search Engine ID or raise a config error."""
    cse_id = _read_secret("GOOGLE_CSE_ID") or _read_secret("GOOGLE_SEARCH_ENGINE_ID")
    if cse_id:
        return cse_id.strip()

    raise RAGConfigurationError(
        "Missing GOOGLE_CSE_ID. Add your Programmable Search Engine ID to .env or Streamlit secrets."
    )


def get_serpapi_api_key() -> str:
    """Return the SerpAPI key or raise a helpful configuration error."""
    api_key = _read_secret("SERPAPI_KEY") or _read_secret("SERPAPI_API_KEY")
    if api_key:
        return api_key.strip()

    raise RAGConfigurationError(
        "Missing SERPAPI_KEY. Add it to your .env file or Streamlit secrets."
    )


def get_serpapi_location() -> str:
    """Return SerpAPI location (used for region targeting)."""
    return (_read_secret("SERPAPI_LOCATION") or SERPAPI_LOCATION_DEFAULT).strip()


def get_serpapi_hl() -> str:
    """Return SerpAPI language (hl)."""
    return (_read_secret("SERPAPI_HL") or SERPAPI_HL_DEFAULT).strip()


def get_serpapi_gl() -> str:
    """Return SerpAPI country code (gl)."""
    return (_read_secret("SERPAPI_GL") or SERPAPI_GL_DEFAULT).strip()


def get_serpapi_google_domain() -> str:
    """Return the google_domain parameter for SerpAPI."""
    return (
        _read_secret("SERPAPI_GOOGLE_DOMAIN") or SERPAPI_GOOGLE_DOMAIN_DEFAULT
    ).strip()


def build_groq_llm(
    model_name: str = DEFAULT_GROQ_MODEL,
    temperature: float = 0.0,
) -> ChatGroq:
    """Create a deterministic Groq-backed chat model for retrieval tasks."""
    return ChatGroq(
        model=model_name,
        temperature=temperature,
        groq_api_key=get_groq_api_key(),
        max_retries=2,
    )


def _fallback_query(cleaned_text: str, max_terms: int = 12) -> str:
    """Fallback search query when the LLM output is empty or malformed."""
    tokens = cleaned_text.split()
    if not tokens:
        return ""
    return " ".join(tokens[:max_terms])


def _fallback_claim(raw_text: str, max_words: int = 28) -> str:
    """Choose a reasonable claim candidate if the LLM extraction fails."""
    text = re.sub(r"\s+", " ", str(raw_text)).strip()
    if not text:
        return ""

    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    for sentence in sentences:
        sentence = sentence.strip().strip("\"'")
        if sentence:
            words = sentence.split()
            if len(words) > max_words:
                sentence = " ".join(words[:max_words])
            return sentence

    words = text.split()
    return " ".join(words[:max_words])


def _sanitize_search_query(query: str, max_terms: int = 12) -> str:
    """Normalize the model output into a compact web-search query."""
    query = str(query).strip()
    query = re.sub(r"^search query\s*:\s*", "", query, flags=re.IGNORECASE)
    query = query.replace("\n", " ")
    query = query.strip(" `\"'")
    query = re.sub(r"\s+", " ", query).strip()

    terms = query.split()
    if len(terms) > max_terms:
        query = " ".join(terms[:max_terms])

    return query


def _sanitize_claim(claim: str, max_words: int = 32) -> str:
    """Normalize the extracted claim into one concise sentence."""
    claim = str(claim).strip()
    claim = re.sub(r"^claim\s*:\s*", "", claim, flags=re.IGNORECASE)
    claim = claim.replace("\n", " ")
    claim = claim.strip(" `\"'")
    claim = re.sub(r"\s+", " ", claim).strip()

    words = claim.split()
    if len(words) > max_words:
        claim = " ".join(words[:max_words])

    return claim


def _extract_json_object(text: str) -> str:
    """Pull the first JSON object out of an LLM response."""
    text = str(text).strip()
    fenced_match = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced_match:
        return fenced_match.group(1)

    direct_match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if direct_match:
        return direct_match.group(1)

    return text


def _safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    """Attempt to parse a JSON object from text."""
    try:
        payload = json.loads(_extract_json_object(text))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def extract_claim_and_query(
    raw_text: str,
    llm: Optional[ChatGroq] = None,
    model_name: str = DEFAULT_GROQ_MODEL,
) -> Dict[str, Any]:
    """
    Extract one primary verifiable claim and a concise search query.

    Returning both values from one LLM call keeps retrieval aligned with the
    exact claim being verified.
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("Input text is empty. Please provide a claim to analyze.")

    processor = DataProcessor()
    cleaned_text = processor.clean_text(raw_text)
    fallback_query = _fallback_query(cleaned_text)
    fallback_claim = _fallback_claim(raw_text)

    if not fallback_query:
        raise ValueError("Could not extract searchable terms from the provided text.")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You prepare news claims for live web verification.\n"
                    "Return ONLY valid JSON with this exact schema:\n"
                    "{{\n"
                    '  "claim": "one precise factual claim",\n'
                    '  "query": "a concise web search query"\n'
                    "}}\n"
                    "\n"
                    "Rules:\n"
                    "1. Extract one primary verifiable claim only.\n"
                    "2. Keep the claim concrete and factual.\n"
                    "3. Preserve important names, dates, places, and numbers.\n"
                    "4. Keep the query under 12 words.\n"
                    "5. Do not explain your answer.\n"
                    "6. Do not include markdown or extra keys."
                ),
            ),
            (
                "human",
                (
                    "Original user text:\n{raw_text}\n\n"
                    "Lightly cleaned text:\n{cleaned_text}\n\n"
                    "Extract the best single claim to verify and the best search query."
                ),
            ),
        ]
    )

    llm = llm or build_groq_llm(model_name=model_name)
    chain = prompt | llm
    response = chain.invoke(
        {
            "raw_text": raw_text[:4000],
            "cleaned_text": cleaned_text[:2500],
        }
    )

    payload = _safe_json_loads(response.content) or {}
    claim = _sanitize_claim(payload.get("claim", "")) or fallback_claim
    query = _sanitize_search_query(payload.get("query", "")) or _sanitize_search_query(claim) or fallback_query

    return {
        "claim": claim,
        "query": query,
        "parse_ok": bool(payload),
        "raw_response": str(response.content).strip(),
    }


def generate_search_query(
    raw_text: str,
    llm: Optional[ChatGroq] = None,
    model_name: str = DEFAULT_GROQ_MODEL,
) -> str:
    """Backward-compatible wrapper around claim/query extraction."""
    return extract_claim_and_query(raw_text, llm=llm, model_name=model_name)["query"]


def _fetch_url_content(url: str, timeout: int = DEFAULT_FETCH_TIMEOUT) -> str:
    """Fetch a small amount of article text for stronger grounding than snippets."""
    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers=REQUEST_HEADERS,
            allow_redirects=True,
        )
        response.raise_for_status()
    except requests.RequestException:
        return ""

    content_type = response.headers.get("Content-Type", "").lower()
    if "text/html" not in content_type:
        return ""

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg", "header", "footer", "nav", "form"]):
        tag.decompose()

    root = soup.find("article") or soup.find("main") or soup.body or soup
    text = " ".join(root.stripped_strings)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_PAGE_CHARS]


def _enrich_search_results(search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach fetched article text when it is available."""
    enriched_results: List[Dict[str, Any]] = []

    for result in search_results:
        content = _fetch_url_content(result["url"])
        enriched_results.append(
            {
                **result,
                "content": content,
            }
        )

    return enriched_results


def _fallback_duckduckgo_search(query: str, max_results: int) -> List[Dict[str, Any]]:
    """Fallback search using DuckDuckGo HTML when Google API fails."""
    try:
        url = "https://html.duckduckgo.com/html/"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        }
        response = requests.get(url, params={"q": query}, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        results = []
        for result in soup.find_all("div", class_="result", limit=max_results):
            title_tag = result.find("a", class_="result__title")
            snippet_tag = result.find("a", class_="result__snippet")
            url_tag = result.find("a", class_="result__url")
            
            if title_tag and url_tag:
                title = title_tag.get_text(strip=True)
                snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
                url_str = url_tag.get("href", "")
                
                if "uddg=" in url_str:
                    import urllib.parse
                    parsed = urllib.parse.urlparse(url_str)
                    qs = urllib.parse.parse_qs(parsed.query)
                    if "uddg" in qs:
                        url_str = qs["uddg"][0]
                
                if url_str and not url_str.startswith("/"):
                    results.append({
                        "id": len(results) + 1,
                        "title": title,
                        "snippet": snippet,
                        "url": url_str
                    })
        return results
    except Exception as e:
        print(f"DuckDuckGo fallback failed: {e}")
        return []


def search_web(
    query: str,
    max_results: int = DEFAULT_MAX_SEARCH_RESULTS,
    enrich: bool = True,
) -> List[Dict[str, Any]]:
    """
    Retrieve live web results for the generated query using SerpAPI (primary).

    Results are normalized, deduplicated, and optionally enriched with article
    text so the verifier gets more than just short snippets.
    """
    if not query or not query.strip():
        raise ValueError("Search query is empty. Cannot perform live retrieval.")

    result_count = max(1, min(max_results, 10))
    normalized_results: List[Dict[str, Any]] = []
    seen_urls = set()

    try:
        try:
            import serpapi  # type: ignore
        except ImportError as exc:
            raise RAGConfigurationError(
                "Missing 'serpapi' package. Install it with: pip install serpapi"
            ) from exc

        client = serpapi.Client(
            api_key=get_serpapi_api_key(),
            timeout=DEFAULT_FETCH_TIMEOUT,
        )

        serpapi_location = get_serpapi_location()
        serpapi_hl = get_serpapi_hl()
        serpapi_gl = get_serpapi_gl()
        serpapi_google_domain = get_serpapi_google_domain()

        payload = client.search(
            {
                "engine": _read_secret("SERPAPI_ENGINE") or SERPAPI_ENGINE_DEFAULT,
                "q": query,
                "location": serpapi_location,
                "hl": serpapi_hl,
                "gl": serpapi_gl,
                "google_domain": serpapi_google_domain,
            }
        )

        organic_results = payload.get("organic_results") or []
        if not organic_results:
            # Some endpoints can respond with different keys.
            organic_results = payload.get("news_results") or payload.get("results") or []

        for item in organic_results:
            url = str(item.get("link") or item.get("url") or "").strip()
            title = str(item.get("title") or "Untitled Source").strip()
            snippet = str(item.get("snippet") or item.get("description") or "").strip()

            if not url or url in seen_urls:
                continue

            seen_urls.add(url)
            normalized_results.append(
                {
                    "id": len(normalized_results) + 1,
                    "title": title,
                    "snippet": snippet,
                    "url": url,
                }
            )

            if len(normalized_results) >= result_count:
                break

        # If SerpAPI succeeded but returned no results, fall back to HTML scraping.
        if not normalized_results:
            print("SerpAPI returned no results. Falling back to DuckDuckGo...")
            fallback_results = _fallback_duckduckgo_search(query, result_count)
            if enrich and fallback_results:
                fallback_results = _enrich_search_results(fallback_results)
            return fallback_results

    except RAGConfigurationError:
        # Configuration mistakes should be surfaced to the user.
        raise
    except Exception as exc:
        print(f"SerpAPI search failed: {exc}. Falling back to DuckDuckGo...")
        fallback_results = _fallback_duckduckgo_search(query, result_count)
        if enrich and fallback_results:
            fallback_results = _enrich_search_results(fallback_results)
        return fallback_results

    if enrich and normalized_results:
        normalized_results = _enrich_search_results(normalized_results)

    return normalized_results


def _format_search_context(search_results: List[Dict[str, Any]]) -> str:
    """Convert search results into a compact prompt-friendly evidence block."""
    blocks = []
    for result in search_results:
        content = str(result.get("content", "")).strip()
        evidence = content or str(result.get("snippet", "")).strip()
        blocks.append(
            (
                f"[Source {result['id']}]\n"
                f"Title: {result['title']}\n"
                f"Snippet: {result['snippet']}\n"
                f"Content Excerpt: {evidence}\n"
                f"URL: {result['url']}"
            )
        )
    return "\n\n".join(blocks)


def _normalize_verdict(value: str) -> str:
    """Constrain model verdicts to the supported label set."""
    value = str(value).strip().lower()
    mapping = {
        "true": "True",
        "supported": "True",
        "verified": "True",
        "real": "True",
        "false": "False",
        "refuted": "False",
        "contradicted": "False",
        "fake": "False",
        "unverified": "Unverified",
        "uncertain": "Unverified",
        "unknown": "Unverified",
        "mixed": "Unverified",
    }
    return mapping.get(value, "Unverified")


def _infer_verdict_from_text(text: str) -> str:
    """Best-effort verdict inference when the JSON format is broken."""
    lower_text = str(text).lower()

    false_patterns = [
        r'"verdict"\s*:\s*"false"',
        r"\bverdict\s*[:\-]\s*false\b",
        r"\bthe claim is false\b",
        r"\bclaim is refuted\b",
    ]
    for pattern in false_patterns:
        if re.search(pattern, lower_text):
            return "False"

    true_patterns = [
        r'"verdict"\s*:\s*"true"',
        r"\bverdict\s*[:\-]\s*true\b",
        r"\bthe claim is true\b",
        r"\bclaim is supported\b",
    ]
    for pattern in true_patterns:
        if re.search(pattern, lower_text):
            return "True"

    return "Unverified"


def _parse_source_ids(value: Any) -> List[int]:
    """Normalize source identifiers into a clean integer list."""
    if isinstance(value, list):
        candidates = value
    else:
        candidates = re.findall(r"\d+", str(value))

    normalized_ids = []
    for source_id in candidates:
        try:
            normalized_ids.append(int(source_id))
        except (TypeError, ValueError):
            continue

    return normalized_ids


def _parse_verification_response(response_text: str) -> Dict[str, Any]:
    """Parse the JSON response and preserve raw output for debugging."""
    default_payload: Dict[str, Any] = {
        "verdict": "Unverified",
        "reasoning": "The verifier did not return valid JSON. Review the raw model output.",
        "source_ids": [],
        "parse_ok": False,
        "raw_response": str(response_text).strip(),
    }

    payload = _safe_json_loads(response_text)
    if not payload:
        default_payload["verdict"] = _infer_verdict_from_text(response_text)
        default_payload["reasoning"] = str(response_text).strip() or default_payload["reasoning"]
        default_payload["source_ids"] = _parse_source_ids(response_text)
        return default_payload

    verdict = _normalize_verdict(payload.get("verdict", "Unverified"))
    reasoning = str(payload.get("reasoning", "")).strip() or default_payload["reasoning"]
    source_ids = _parse_source_ids(payload.get("source_ids", []))

    return {
        "verdict": verdict,
        "reasoning": reasoning,
        "source_ids": source_ids,
        "parse_ok": True,
        "raw_response": str(response_text).strip(),
    }


def verify_claim_with_context(
    claim_text: str,
    search_results: List[Dict[str, Any]],
    raw_text: Optional[str] = None,
    llm: Optional[ChatGroq] = None,
    model_name: str = DEFAULT_GROQ_MODEL,
) -> Dict[str, Any]:
    """
    Ask the LLM to verify one extracted claim using retrieved evidence only.

    The verifier is allowed to return True or False when the retrieved sources
    clearly align, and it preserves raw model output for easier debugging.
    """
    if not claim_text or not claim_text.strip():
        raise ValueError("Claim text is empty. Please provide a claim to verify.")

    if not search_results:
        return {
            "verdict": "Unverified",
            "reasoning": (
                "No live search evidence was retrieved, so the claim cannot be "
                "verified reliably."
            ),
            "source_ids": [],
            "sources": [],
            "parse_ok": True,
            "raw_response": "",
        }

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a disciplined fact-verification assistant.\n"
                    "You MUST rely only on the retrieved search context provided.\n"
                    "Do not use outside knowledge.\n"
                    "Do not guess.\n"
                    "\n"
                    "Verdict policy:\n"
                    "- Return 'True' when the retrieved evidence directly supports the claim.\n"
                    "- Return 'False' when the retrieved evidence directly contradicts or refutes the claim.\n"
                    "- Return 'Unverified' only when the evidence is weak, mixed, off-topic, or insufficient.\n"
                    "\n"
                    "Return ONLY valid JSON with this exact schema:\n"
                    "{{\n"
                    '  "verdict": "True" | "False" | "Unverified",\n'
                    '  "reasoning": "2-5 sentences grounded only in the retrieved context",\n'
                    '  "source_ids": [1, 2, 3]\n'
                    "}}\n"
                    "\n"
                    "Rules:\n"
                    "1. Cite only source IDs that directly support your decision.\n"
                    "2. Never invent source IDs.\n"
                    "3. If the query evidence is about a different claim, return 'Unverified'.\n"
                    "4. Do not output markdown or any text outside the JSON."
                ),
            ),
            (
                "human",
                (
                    "Primary claim to verify:\n{claim_text}\n\n"
                    "Original user submission for context:\n{raw_text}\n\n"
                    "Retrieved context:\n{search_context}\n\n"
                    "Decide whether the primary claim is supported, contradicted, or still unverified."
                ),
            ),
        ]
    )

    llm = llm or build_groq_llm(model_name=model_name)
    chain = prompt | llm
    response = chain.invoke(
        {
            "claim_text": claim_text[:2000],
            "raw_text": (raw_text or claim_text)[:2500],
            "search_context": _format_search_context(search_results)[:MAX_CONTEXT_CHARS],
        }
    )

    parsed = _parse_verification_response(response.content)
    valid_ids = {result["id"] for result in search_results}
    cited_ids = [source_id for source_id in parsed["source_ids"] if source_id in valid_ids]
    cited_sources = [result for result in search_results if result["id"] in cited_ids]

    return {
        "claim": claim_text,
        "verdict": parsed["verdict"],
        "reasoning": parsed["reasoning"],
        "source_ids": cited_ids,
        "sources": cited_sources,
        "parse_ok": parsed["parse_ok"],
        "raw_response": parsed["raw_response"],
    }


def deep_verify_claim(
    raw_text: str,
    model_name: str = DEFAULT_GROQ_MODEL,
    max_results: int = DEFAULT_MAX_SEARCH_RESULTS,
) -> Dict[str, Any]:
    """
    Run the complete Layer 2 pipeline:
    claim extraction -> retrieval -> grounded verification.
    """
    llm = build_groq_llm(model_name=model_name)
    extraction = extract_claim_and_query(raw_text, llm=llm, model_name=model_name)
    search_results = search_web(
        query=extraction["query"],
        max_results=max_results,
        enrich=True,
    )
    verification = verify_claim_with_context(
        claim_text=extraction["claim"],
        raw_text=raw_text,
        search_results=search_results,
        llm=llm,
        model_name=model_name,
    )

    return {
        "claim": extraction["claim"],
        "query": extraction["query"],
        "extraction_parse_ok": extraction["parse_ok"],
        "extraction_raw_response": extraction["raw_response"],
        "search_results": search_results,
        **verification,
    }
