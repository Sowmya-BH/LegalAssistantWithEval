/**
 * Typed fetch wrappers — one per endpoint in api/query_routes.py and
 * api/ingest_routes.py. Same-origin (the frontend is served by the same
 * FastAPI process — see api/main.py's StaticFiles mount), so no base URL
 * is needed.
 */

import type {
  AnswerDecisionRequest,
  CollectionsResponse,
  EvidenceDecisionRequest,
  IngestJobResponse,
  QueryListItem,
  QueryStartRequest,
  QueryStateResponse,
} from "./types.js";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
  return res.json() as Promise<T>;
}

function json(body: unknown): RequestInit {
  return { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
}

export const api = {
  startQuery: (req: QueryStartRequest) => request<QueryStateResponse>("/api/query/start", json(req)),

  listThreads: () => request<QueryListItem[]>("/api/query"),

  getQuery: (threadId: string) => request<QueryStateResponse>(`/api/query/${threadId}`),

  submitEvidenceDecision: (threadId: string, decision: EvidenceDecisionRequest) =>
    request<QueryStateResponse>(`/api/query/${threadId}/evidence-decision`, json(decision)),

  submitAnswerDecision: (threadId: string, decision: AnswerDecisionRequest) =>
    request<QueryStateResponse>(`/api/query/${threadId}/answer-decision`, json(decision)),

  startIngest: (file: File, collectionName: string | null) => {
    const form = new FormData();
    form.append("file", file);
    const qs = collectionName ? `?collection_name=${encodeURIComponent(collectionName)}` : "";
    return request<IngestJobResponse>(`/api/ingest${qs}`, { method: "POST", body: form });
  },

  getIngestJob: (jobId: string) => request<IngestJobResponse>(`/api/ingest/${jobId}`),

  listCollections: () => request<CollectionsResponse>("/api/ingest/meta/collections"),
};
