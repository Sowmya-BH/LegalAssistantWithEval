/**
 * Entry point — wires DOM events for the three tabs (Ask / Ingest /
 * Threads). Compiled to dist/main.js and loaded by index.html as an ES
 * module. No framework: direct DOM + the typed api/render helpers.
 */

import { api } from "./api.js";
import { escapeHtml, renderQueryState } from "./render.js";
import type { AnswerAction, IngestJobResponse, QueryStateResponse } from "./types.js";

let currentThreadId: string | null = null;

function el<T extends HTMLElement>(id: string): T {
  const found = document.getElementById(id);
  if (!found) throw new Error(`Missing element #${id}`);
  return found as T;
}

// ---------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------

function initTabs(): void {
  document.querySelectorAll<HTMLButtonElement>(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      const tab = btn.dataset.tab!;
      el(`panel-${tab}`).classList.add("active");
      if (tab === "threads") void loadThreads();
      if (tab === "ask") void loadCollections();
    });
  });
}

// ---------------------------------------------------------------------------
// Ask
// ---------------------------------------------------------------------------

async function loadCollections(): Promise<void> {
  const sel = el<HTMLSelectElement>("ask-collection");
  try {
    const { collections } = await api.listCollections();
    sel.innerHTML = collections.length
      ? collections.map((c) => `<option value="${c}">${c}</option>`).join("")
      : '<option value="">(none — ingest a PDF first)</option>';
  } catch {
    sel.innerHTML = '<option value="">(failed to load collections)</option>';
  }
}

function renderAndWireQueryState(state: QueryStateResponse): void {
  el("ask-result").innerHTML = renderQueryState(state);

  document.getElementById("ev-approve")?.addEventListener("click", () => void submitEvidenceDecision(true));
  document.getElementById("ev-reject")?.addEventListener("click", () => void submitEvidenceDecision(false));
  document.getElementById("ans-approve")?.addEventListener("click", () => void submitAnswerDecision("approve"));
  document.getElementById("ans-revise")?.addEventListener("click", () => void submitAnswerDecision("revise"));
  document.getElementById("ans-reject")?.addEventListener("click", () => void submitAnswerDecision("reject"));
}

async function submitEvidenceDecision(proceed: boolean): Promise<void> {
  if (!currentThreadId) return;
  const state = await api.submitEvidenceDecision(currentThreadId, { proceed, reviewer: "web-ui" });
  renderAndWireQueryState(state);
}

async function submitAnswerDecision(action: AnswerAction): Promise<void> {
  if (!currentThreadId) return;
  const comments = (document.getElementById("answer-comments") as HTMLTextAreaElement | null)?.value || null;
  const state = await api.submitAnswerDecision(currentThreadId, { action, reviewer: "web-ui", comments });
  renderAndWireQueryState(state);
}

function initAsk(): void {
  el("ask-submit").addEventListener("click", () => {
    void (async () => {
      const question = el<HTMLTextAreaElement>("ask-question").value.trim();
      const collection_name = el<HTMLSelectElement>("ask-collection").value;
      const errEl = el("ask-error");
      errEl.textContent = "";

      if (!question || !collection_name) {
        errEl.textContent = "Question and collection are required.";
        return;
      }
      try {
        const state = await api.startQuery({ question, collection_name });
        currentThreadId = state.thread_id;
        renderAndWireQueryState(state);
      } catch (e) {
        errEl.textContent = e instanceof Error ? e.message : String(e);
      }
    })();
  });
}

// ---------------------------------------------------------------------------
// Ingest
// ---------------------------------------------------------------------------

function initIngest(): void {
  el("ingest-submit").addEventListener("click", () => {
    void (async () => {
      const fileInput = el<HTMLInputElement>("ingest-file");
      const collection = el<HTMLInputElement>("ingest-collection").value.trim();
      const errEl = el("ingest-error");
      errEl.textContent = "";

      const file = fileInput.files?.[0];
      if (!file) {
        errEl.textContent = "Choose a PDF first.";
        return;
      }
      try {
        const job = await api.startIngest(file, collection || null);
        pollIngestJob(job.job_id);
      } catch (e) {
        errEl.textContent = e instanceof Error ? e.message : String(e);
      }
    })();
  });
}

function pollIngestJob(jobId: string): void {
  const resultEl = el("ingest-result");
  resultEl.style.display = "block";
  resultEl.innerHTML = `<p class="muted">Job <code>${jobId}</code>: pending...</p>`;

  const interval = window.setInterval(() => {
    void (async () => {
      const job: IngestJobResponse = await api.getIngestJob(jobId);
      if (job.status === "running" || job.status === "pending") {
        resultEl.innerHTML = `<p class="muted">Job <code>${jobId}</code>: ${job.status}...</p>`;
        return;
      }
      window.clearInterval(interval);
      if (job.status === "done") {
        resultEl.innerHTML = `<p><b>Done.</b> Collection: <code>${escapeHtml(job.collection_name ?? "")}</code></p>
          <details class="collapsible"><summary>&#9654; Ingestion result details</summary>
          <div class="content"><pre>${escapeHtml(JSON.stringify(job.result, null, 2))}</pre></div></details>`;
        void loadCollections();
      } else {
        resultEl.innerHTML = `<p class="error">Failed: ${escapeHtml(job.error ?? "unknown error")}</p>`;
      }
    })();
  }, 2000);
}

// ---------------------------------------------------------------------------
// Threads
// ---------------------------------------------------------------------------

async function loadThreads(): Promise<void> {
  const tbody = document.querySelector<HTMLTableSectionElement>("#threads-table tbody")!;
  const empty = el("threads-empty");
  const threads = await api.listThreads();
  tbody.innerHTML = "";
  empty.style.display = threads.length ? "none" : "block";

  for (const t of threads) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${escapeHtml(t.question)}</td><td>${escapeHtml(t.collection_name)}</td>
      <td><span class="status-pill">${escapeHtml(t.status)}</span></td>
      <td>${new Date(t.created_at).toLocaleString()}</td>
      <td><button class="secondary" data-thread="${t.thread_id}">Open</button></td>`;
    tbody.appendChild(tr);
  }

  tbody.querySelectorAll<HTMLButtonElement>("button[data-thread]").forEach((btn) => {
    btn.addEventListener("click", () => {
      void (async () => {
        currentThreadId = btn.dataset.thread!;
        document.querySelector<HTMLButtonElement>('.tab-btn[data-tab="ask"]')!.click();
        const state = await api.getQuery(currentThreadId);
        renderAndWireQueryState(state);
      })();
    });
  });
}

function initThreads(): void {
  el("threads-refresh").addEventListener("click", () => void loadThreads());
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

initTabs();
initAsk();
initIngest();
initThreads();
void loadCollections();
