export type ImageTask = "text-to-image" | "inpaint" | "outpaint";
export type TaskKind = ImageTask | "select" | "load-model";
export type JobState = "queued" | "running" | "cancelling" | "cancelled" | "completed" | "failed";
export type RuntimeMode = "safe" | "balanced" | "performance";

export interface ModelInfo {
  name: string;
  display_name: string;
  model_id: string;
  capabilities: TaskKind[];
  loaded: boolean;
  loaded_pipelines: string[];
}

export interface GenerationRecord {
  id: number;
  task: ImageTask;
  prompt: string;
  negative_prompt: string | null;
  model_name: string;
  runtime_mode: RuntimeMode;
  seed: number;
  steps: number;
  guidance: number;
  width: number | null;
  height: number | null;
  strength: number | null;
  edit_operation: "replace" | "remove" | null;
  blend_feather: number | null;
  context_padding: number | null;
  scene_context: string | null;
  input_image_path: string | null;
  mask_image_path: string | null;
  output_image_path: string;
  output_image_url: string;
  created_at: string;
}

export interface JobResponse {
  id: string;
  kind: TaskKind;
  status: JobState;
}

export interface JobStatus extends JobResponse {
  progress: number;
  error: string | null;
  generation_id: number | null;
  generation_ids: number[];
  mask_image_url?: string | null;
}

export interface SystemInfo {
  hardware_profile: string;
  torch_available: boolean;
  torch_version?: string;
  preferred_device?: string;
  selected_device?: string;
  preferred_dtype?: string;
  enable_tf32?: boolean;
  enable_cpu_offload?: boolean;
  allow_cpu_generation?: boolean;
  runtime_mode?: RuntimeMode;
  gpu_resident_max_pixels?: number;
  max_loaded_pipelines?: number;
  attention_backend?: string;
  cuda_available?: boolean;
  cuda_device?: string;
  cuda_version?: string;
  vram_gb?: number;
  cuda_memory_fraction?: number;
  setup_hint?: string;
  error?: string;
}

export interface TextToImageRequest {
  prompt: string;
  negative_prompt?: string;
  model_name: string;
  runtime_mode: RuntimeMode;
  seed?: number;
  steps: number;
  guidance: number;
  width: number;
  height: number;
  count: number;
}

export interface DocumentRecord {
  id: number;
  name: string;
  canvas_width: number;
  canvas_height: number;
  color_profile: string;
  current_revision_id: number | null;
  current_image_url: string | null;
  can_undo: boolean;
  can_redo: boolean;
  created_at: string;
  updated_at: string;
}

export interface AlternativeRecord {
  id: number;
  revision_id: number;
  generation_id: number | null;
  seed: number;
  model_name: string;
  output_image_path: string;
  output_image_url: string;
  metadata: Record<string, unknown>;
  selected: boolean;
}

export interface RevisionRecord {
  id: number;
  document_id: number;
  parent_revision_id: number | null;
  operation_type: string;
  composite_image_path: string;
  composite_image_url: string;
  mask_image_path: string | null;
  operation_parameters: Record<string, unknown>;
  selected_alternative_id: number | null;
  accepted: boolean;
  created_at: string;
  alternatives: AlternativeRecord[];
}

export interface LoadModelRequest {
  model_name: string;
  task: "text-to-image" | "inpaint";
  runtime_mode: RuntimeMode;
  width: number;
  height: number;
}

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

export function fileUrl(path: string): string {
  // Retire cached image responses from the pre-CORS editor.
  return `${API_BASE}${path}${path.includes("?") ? "&" : "?"}v=2`;
}

export async function listModels(): Promise<ModelInfo[]> {
  return request("/api/models");
}

export async function listHistory(): Promise<GenerationRecord[]> {
  return request("/api/history");
}

export async function getSystemInfo(): Promise<SystemInfo> {
  return request("/api/system");
}

export async function getGeneration(id: number): Promise<GenerationRecord> {
  return request(`/api/history/${id}`);
}

export async function generateImage(payload: TextToImageRequest): Promise<JobResponse> {
  return request("/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function loadModel(payload: LoadModelRequest): Promise<JobResponse> {
  return request("/api/models/load", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function inpaintImage(form: FormData): Promise<JobResponse> {
  return request("/api/inpaint", {
    method: "POST",
    body: form,
  });
}

export interface SelectionPoint { x: number; y: number; label: 0 | 1 }
export interface Expansion { left: number; top: number; right: number; bottom: number }

export async function outpaintImage(image: File, parameters: Omit<TextToImageRequest, "width" | "height"> & Expansion): Promise<JobResponse> {
  const form = new FormData();
  form.set("image", image);
  form.set("parameters", JSON.stringify(parameters));
  return request("/api/outpaint", { method: "POST", body: form });
}

export async function selectObject(image: File, points: SelectionPoint[]): Promise<JobResponse> {
  const form = new FormData();
  form.set("image", image);
  form.set("parameters", JSON.stringify({ points }));
  return request("/api/select", { method: "POST", body: form });
}

export async function getDocument(id: number): Promise<DocumentRecord> {
  return request(`/api/v2/documents/${id}`);
}

export async function getJob(id: string): Promise<JobStatus> {
  return request(`/api/jobs/${id}`);
}

export async function cancelJob(id: string): Promise<JobStatus> {
  return request(`/api/jobs/${id}/cancel`, { method: "POST" });
}

export async function listDocuments(): Promise<DocumentRecord[]> {
  return request("/api/v2/documents");
}

export async function createDocument(image: File, name = image.name): Promise<DocumentRecord> {
  const form = new FormData();
  form.set("name", name);
  form.set("image", image);
  return request("/api/v2/documents", { method: "POST", body: form });
}

export async function importGeneration(generationId: number): Promise<DocumentRecord> {
  return request(`/api/v2/documents/import/${generationId}`, { method: "POST" });
}

export async function updateDocument(documentId: number, name: string): Promise<DocumentRecord> {
  return request(`/api/v2/documents/${documentId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
}

export async function proposeAlternatives(
  documentId: number,
  generationIds: number[],
  operationType: "generate" | "inpaint" | "outpaint",
  baseRevisionId?: number,
): Promise<RevisionRecord> {
  return request(`/api/v2/documents/${documentId}/alternatives`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ generation_ids: generationIds, operation_type: operationType, base_revision_id: baseRevisionId }),
  });
}

export async function acceptAlternative(
  documentId: number,
  revisionId: number,
  alternativeId: number,
): Promise<DocumentRecord> {
  return request(
    `/api/v2/documents/${documentId}/revisions/${revisionId}/alternatives/${alternativeId}/accept`,
    { method: "POST" },
  );
}

export async function undoDocument(documentId: number): Promise<DocumentRecord> {
  return request(`/api/v2/documents/${documentId}/undo`, { method: "POST" });
}

export async function redoDocument(documentId: number): Promise<DocumentRecord> {
  return request(`/api/v2/documents/${documentId}/redo`, { method: "POST" });
}

export class ApiError extends Error {
  constructor(message: string, readonly status: number) { super(message); }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);
  if (!response.ok) {
    const text = await response.text();
    let message = text;
    try {
      const payload: unknown = JSON.parse(text);
      if (typeof payload === "object" && payload !== null && "detail" in payload && typeof payload.detail === "string") message = payload.detail;
    } catch { /* Plain-text server errors remain useful. */ }
    throw new ApiError(message || `Request failed: ${response.status}`, response.status);
  }
  return response.json();
}
