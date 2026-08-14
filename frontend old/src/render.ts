/**
 * Rendering rules (per spec): the Answer text is the only thing that stays
 * permanently visible. Citations/risk/uncertainty and all retrieval/
 * reasoning internals render inside <details> — collapsed by default.
 */

import type { QueryStateResponse, RetrievedChunk, TechnicalDetails } from "./types.js";

export function escapeHtml(s: string | null | undefined): string {
  const map: Record<string, string> = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
  return (s ?? "").replace(/[&<>"']/g, (m) => map[m]);
}

function renderChunk(c: RetrievedChunk): string {
  const score = c.score !== null && c.score !== undefined ? c.score.toFixed(3) : "?";
  return `<div class="chunk">
    <div class="meta">${escapeHtml(c.document_name ?? "?")} · p.${c.page_start ?? "?"} · score=${score}</div>
    ${escapeHtml((c.text ?? "").slice(0, 220))}...
  </div>`;
}

/** The ONLY non-collapsible content block — the headline answer text. */
export function renderAnswerOutput(text: string): string {
  return `<div class="output-card">
    <h3>Answer</h3>
    <div class="output-text">${escapeHtml(text)}</div>
  </div>`;
}

/** Collapsible: citations / risk level / uncertainty flag. */
export function renderCitationsDetails(technical: TechnicalDetails | null | undefined): string {
  if (!technical) return "";
  const citationsHtml = technical.citations.length
    ? `<ul>${technical.citations.map((c) => `<li>${escapeHtml(c)}</li>`).join("")}</ul>`
    : `<p class="muted">(none)</p>`;

  return `<details class="collapsible">
    <summary>&#9654; Citations, risk &amp; uncertainty</summary>
    <div class="content">
      <p class="muted"><b>Risk level:</b> ${escapeHtml(technical.risk_level ?? "n/a")}</p>
      <p class="muted"><b>Has uncertainty:</b> ${technical.has_uncertainty}</p>
      <p class="muted"><b>Citations:</b></p>
      ${citationsHtml}
    </div>
  </details>`;
}

/** Collapsible: route, retrieved chunks, evidence verdict, cypher. */
export function renderTechnicalDetails(technical: TechnicalDetails | null | undefined): string {
  if (!technical) return "";
  const verdict = technical.evidence_verdict ?? {};
  const gaps = (verdict.gaps ?? []).map((g) => `<p class="muted">&nbsp;&nbsp;gap: ${escapeHtml(g)}</p>`).join("");
  const contradictions = (verdict.contradictions ?? [])
    .map((c) => `<p class="muted">&nbsp;&nbsp;contradiction: ${escapeHtml(c)}</p>`)
    .join("");
  const graphHitsHtml = technical.graph_hits.length
    ? `<p class="muted"><b>Graph hits (${technical.graph_hits.length}):</b></p>${technical.graph_hits.map(renderChunk).join("")}`
    : "";
  const cypherHtml = technical.cypher_used
    ? `<p class="muted"><b>Cypher (${escapeHtml(technical.cypher_source ?? "")}):</b></p><pre>${escapeHtml(technical.cypher_used)}</pre>`
    : "";

  return `<details class="collapsible">
    <summary>&#9654; Retrieval &amp; reasoning details</summary>
    <div class="content">
      <p class="muted">Route: <b>${escapeHtml(technical.route ?? "?")}</b>${
    technical.alpha != null ? ` (alpha=${technical.alpha})` : ""
  } · Revisions: ${technical.answer_revision_count}</p>
      ${technical.route_reasoning ? `<p class="muted">${escapeHtml(technical.route_reasoning)}</p>` : ""}
      <p class="muted"><b>Evidence auditor:</b> sufficient=${verdict.sufficient}</p>
      ${gaps}${contradictions}
      <p class="muted"><b>Hybrid hits (${technical.hybrid_hits.length}):</b></p>
      ${technical.hybrid_hits.map(renderChunk).join("")}
      ${graphHitsHtml}
      ${cypherHtml}
    </div>
  </details>`;
}

/** Full render for a QueryStateResponse, dispatched by status. */
export function renderQueryState(state: QueryStateResponse): string {
  const statusPill = `<span class="status-pill">${state.status}</span>`;

  if (state.status === "awaiting_evidence_approval") {
    return `<div class="card" id="evidence-checkpoint">
      <p>${statusPill} Thread <code>${state.thread_id}</code></p>
      <p>Evidence checkpoint reached — review the retrieved evidence before synthesis proceeds.</p>
      <div class="row">
        <button id="ev-approve">Proceed to synthesis</button>
        <button id="ev-reject" class="danger">Stop here</button>
      </div>
      ${renderTechnicalDetails(state.technical)}
    </div>`;
  }

  if (state.status === "awaiting_answer_approval") {
    return `<div>
      <p>${statusPill} Thread <code>${state.thread_id}</code> (draft — pending review)</p>
      ${renderAnswerOutput(state.draft_answer ?? "")}
      <div class="card">
        <label>Feedback (required for "Revise")</label>
        <textarea id="answer-comments"></textarea>
        <div class="row">
          <button id="ans-approve">Approve</button>
          <button id="ans-revise" class="secondary">Revise</button>
          <button id="ans-reject" class="danger">Reject</button>
        </div>
      </div>
      ${renderCitationsDetails(state.technical)}
      ${renderTechnicalDetails(state.technical)}
    </div>`;
  }

  if (state.status === "answered") {
    return `<div>
      <p>${statusPill} Thread <code>${state.thread_id}</code></p>
      ${renderAnswerOutput(state.final_answer ?? "")}
      ${renderCitationsDetails(state.technical)}
      ${renderTechnicalDetails(state.technical)}
    </div>`;
  }

  return `<div class="card"><p>${statusPill} Thread <code>${state.thread_id}</code></p>
    <p class="muted">No approved answer (status: ${state.status}).</p></div>`;
}
