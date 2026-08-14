/**
 * Typed fetch wrappers — one per endpoint in api/query_routes.py and
 * api/ingest_routes.py. Same-origin (the frontend is served by the same
 * FastAPI process — see api/main.py's StaticFiles mount), so no base URL
 * is needed.
 */
async function request(path, init) {
    const res = await fetch(path, init);
    if (!res.ok) {
        const body = await res.text();
        throw new Error(`${res.status} ${res.statusText}: ${body}`);
    }
    return res.json();
}
function json(body) {
    return { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
}
export const api = {
    startQuery: (req) => request("/api/query/start", json(req)),
    listThreads: () => request("/api/query"),
    getQuery: (threadId) => request(`/api/query/${threadId}`),
    submitEvidenceDecision: (threadId, decision) => request(`/api/query/${threadId}/evidence-decision`, json(decision)),
    submitAnswerDecision: (threadId, decision) => request(`/api/query/${threadId}/answer-decision`, json(decision)),
    startIngest: (file, collectionName) => {
        const form = new FormData();
        form.append("file", file);
        const qs = collectionName ? `?collection_name=${encodeURIComponent(collectionName)}` : "";
        return request(`/api/ingest${qs}`, { method: "POST", body: form });
    },
    getIngestJob: (jobId) => request(`/api/ingest/${jobId}`),
    listCollections: () => request("/api/ingest/meta/collections"),
};
