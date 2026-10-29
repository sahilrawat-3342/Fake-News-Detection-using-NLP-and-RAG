const API_BASE = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, "") ?? "";

export type Health = {
  status: string;
  model_online: boolean;
  rag_configured: boolean;
  llm_model: string;
  retrain_available: boolean;
};

export type ConfusionMatrix = {
  true_negatives: number;
  false_positives: number;
  false_negatives: number;
  true_positives: number;
};

export type ModelMetrics = {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  roc_auc?: number;
  confusion_matrix?: ConfusionMatrix;
};

export type MetricsResponse = {
  models: Record<string, { model_name: string; algorithm: string; metrics: ModelMetrics }>;
  layer1: ModelMetrics;
  data_count: number;
  train_time: string | null;
};

export type PredictResponse = {
  label: "real" | "fake";
  prediction: number;
  confidence: number | null;
  real_probability: number | null;
  fake_probability: number | null;
  latency_ms: number;
  clean_text: string;
  token_count: number;
};

export type Source = {
  id: number;
  title: string;
  snippet: string;
  url: string;
  content?: string;
};

export type VerifyResponse = {
  claim: string;
  query: string;
  verdict: "True" | "False" | "Unverified";
  reasoning: string;
  source_ids: number[];
  sources: Source[];
  search_results: Source[];
  parse_ok: boolean;
  extraction_parse_ok: boolean;
  raw_response?: string;
  elapsed_s: number;
};

export type RetrainResponse = { accuracy: number; data_count: number; train_time: string };

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<Health>("/api/health"),
  metrics: () => request<MetricsResponse>("/api/metrics"),
  predict: (text: string) => request<PredictResponse>("/api/predict", { method: "POST", body: JSON.stringify({ text }) }),
  verify: (text: string) => request<VerifyResponse>("/api/verify", { method: "POST", body: JSON.stringify({ text }) }),
  retrain: () => request<RetrainResponse>("/api/retrain", { method: "POST" }),
};

/** Snapshot of performance_metrics.json used until the API responds. */
export const FALLBACK_METRICS: MetricsResponse = {
  data_count: 44898,
  train_time: null,
  layer1: {
    accuracy: 0.9853,
    precision: 0.9806,
    recall: 0.9926,
    f1_score: 0.9866,
    roc_auc: 0.9985,
    confusion_matrix: { true_negatives: 3397, false_positives: 82, false_negatives: 31, true_positives: 4153 },
  },
  models: {
    truthlens_layer1: {
      model_name: "TruthLens Layer 1",
      algorithm: "Logistic Regression + TF-IDF",
      metrics: { accuracy: 0.9853, precision: 0.9806, recall: 0.9926, f1_score: 0.9866 },
    },
    svm_baseline: {
      model_name: "SVM Baseline",
      algorithm: "Linear SVM + TF-IDF",
      metrics: { accuracy: 0.9909, precision: 0.9881, recall: 0.9952, f1_score: 0.9917 },
    },
    naive_bayes_baseline: {
      model_name: "Naive Bayes Baseline",
      algorithm: "Multinomial Naive Bayes + TF-IDF",
      metrics: { accuracy: 0.9485, precision: 0.9334, recall: 0.9751, f1_score: 0.9538 },
    },
  },
};
