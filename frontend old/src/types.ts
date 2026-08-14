/**
 * TypeScript mirrors of api/schemas.py — field names/types match the
 * UNMODIFIED LegalAgentState/SynthesizedAnswer shape exactly (plain string
 * answer, no confidence/evidence/document/source fields — those don't
 * exist in this version of the pipeline).
 */

export interface QueryStartRequest {
  question: string;
  collection_name: string;
  metadata_filter?: Record<string, unknown> | null;
}

export interface EvidenceDecisionRequest {
  proceed: boolean;
  reviewer: string;
  comments?: string | null;
}

export type AnswerAction = "approve" | "revise" | "reject";

export interface AnswerDecisionRequest {
  action: AnswerAction;
  reviewer: string;
  comments?: string | null;
  edited_answer?: string | null;
}

export interface RetrievedChunk {
  text: string;
  document_name?: string | null;
  page_start?: number | null;
  page_end?: number | null;
  section?: string | null;
  score?: number | null;
}

export interface EvidenceVerdict {
  sufficient?: boolean;
  reasoning?: string;
  gaps?: string[];
  contradictions?: string[];
}

export interface TechnicalDetails {
  route?: string | null;
  alpha?: number | null;
  route_reasoning?: string | null;
  hybrid_hits: RetrievedChunk[];
  graph_hits: RetrievedChunk[];
  cypher_used?: string | null;
  cypher_source?: string | null;
  evidence_verdict: EvidenceVerdict;
  answer_revision_count: number;
  citations: string[];
  risk_level?: string | null;
  has_uncertainty: boolean;
}

export type QueryStatus =
  | "awaiting_evidence_approval"
  | "awaiting_answer_approval"
  | "answered"
  | "rejected"
  | "evidence_rejected"
  | "unknown";

export interface QueryStateResponse {
  thread_id: string;
  question: string;
  status: QueryStatus;
  interrupt_type?: string | null;
  interrupt_payload?: Record<string, unknown> | null;
  draft_answer?: string | null;
  final_answer?: string | null;
  technical?: TechnicalDetails | null;
}

export interface QueryListItem {
  thread_id: string;
  question: string;
  collection_name: string;
  status: string;
  created_at: string;
}

export type IngestJobStatus = "pending" | "running" | "done" | "error";

export interface IngestJobResponse {
  job_id: string;
  status: IngestJobStatus;
  filename: string;
  collection_name?: string | null;
  result?: Record<string, unknown> | null;
  error?: string | null;
}

export interface CollectionsResponse {
  collections: string[];
}
