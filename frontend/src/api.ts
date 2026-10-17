export type ImageTask = "text-to-image" | "edit" | "inpaint" | "outpaint";
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
  /** The model this device runs for the Quality or Fast preset: NVFP4 on an RTX 50, BF16 on Apple silicon. */
  preset?: "quality" | "fast" | null;
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
  edit_operation: "replace" | "remove" | "sketch" | "faces" | "enhance" | null;
  blend_feather: number | null;
  context_padding: number | null;
  scene_context: string | null;
  consistency_lora: boolean;
  detail_crop: boolean;
  specialized_outpaint: boolean;
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

export type JobPhase = "loading" | "warming" | "preparing" | "generating" | "finishing";

export interface JobStatus extends JobResponse {
  progress: number;
  error: string | null;
  generation_id: number | null;
  generation_ids: number[];
  mask_image_url?: string | null;
  phase?: JobPhase | null;
  /** The model or component being loaded, when phase is loading or warming. */
  detail?: string | null;
  step?: number | null;
  steps?: number | null;
  /** 1-based proposal being made, out of items. */
  item?: number | null;
  items?: number | null;
  elapsed_seconds?: number | null;
  phase_elapsed_seconds?: number | null;
  eta_seconds?: number | null;
  /** What a queued job waits for, such as the start-up warm-up. */
  waiting_for?: TaskKind | null;
}

export interface EngineStatus {
  /** The start-up job that loads and warms the Quality model and SAM. */
  warm_up: JobStatus | null;
  models: ModelInfo[];
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
  max_loaded_pipelines?: number;
  attention_backend?: string;
  cuda_available?: boolean;
  cuda_device?: string;
  cuda_version?: string;
  vram_gb?: number;
  cuda_memory_fraction?: number;
  /** The GPU or Apple chip, as the engine names it. */
  device_name?: string;
  mps_available?: boolean;
  unified_memory_gb?: number;
  quality_model?: string;
  fast_model?: string;
  /** Loaded while the UI opens: Quality, or Fast on a Mac without room for it. */
  warm_up_model?: string;
  setup_hint?: string;
  error?: string;
}

export interface TextToImageRequest {
  prompt: string;
  model_name: string;
  runtime_mode: RuntimeMode;
  seed?: number;
  steps: number;
  guidance: number;
  width: number;
  height: number;
  count: number;
  /** People or objects to compose into the new image, in prompt order. */
  reference_ids?: number[];
}

export interface DocumentRecord {
  id: number;
  name: string;
  canvas_width: number;
  canvas_height: number;
  color_profile: string;
  current_revision_id: number | null;
  current_image_url: string | null;
  /** The layers that make up the picture, bottom first; the first is usually the Background. */
  layers: LayerRecord[];
  can_undo: boolean;
  can_redo: boolean;
  created_at: string;
  updated_at: string;
}

/** One image layer: centre in canvas pixels, scale (negative flips) and rotation in degrees. */
export interface LayerState {
  id: string;
  name: string;
  image_path: string;
  mask_path: string | null;
  /** A switched-off mask is kept but not applied. */
  mask_enabled: boolean;
  /** AI proposals kept in one layer; `image_path` is the one shown. Empty for ordinary layers. */
  variant_paths: string[];
  x: number;
  y: number;
  scale_x: number;
  scale_y: number;
  rotation: number;
  opacity: number;
  visible: boolean;
  /** The layer an AI edit or Blend in replaced, kept hidden just below; merging this one lets it go. */
  original_id: string | null;
}

export interface LayerRecord extends LayerState {
  width: number;
  height: number;
  image_url: string;
  mask_url: string | null;
  variant_urls: string[];
}

export interface LayerAsset { path: string; url: string; width: number; height: number }

export type ReferenceRole = "subject" | "clothing" | "pose" | "style";

export interface ReferenceRecord {
  id: number;
  /** Null for loose references on the start screen. */
  document_id: number | null;
  name: string;
  role?: ReferenceRole;
  width: number;
  height: number;
  image_url: string;
  subject_url: string | null;
  created_at: string;
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
  /** This revision's layers; an Expand proposal has them moved with the content and its result in one of them. */
  layers: LayerRecord[];
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

export async function getEngine(): Promise<EngineStatus> {
  return request("/api/engine");
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

export type EditParameters = Omit<TextToImageRequest, "width" | "height"> & { consistency_lora?: boolean };

/** Edits the whole picture by instruction, or renders a sketch on it; proposals keep the picture's size. */
export async function editWholeImage(image: File, parameters: EditParameters & { mode?: "edit" | "sketch" }): Promise<JobResponse> {
  const form = new FormData();
  form.set("image", image);
  form.set("parameters", JSON.stringify(parameters));
  return request("/api/edit", { method: "POST", body: form });
}

export async function inpaintImage(form: FormData): Promise<JobResponse> {
  return request("/api/inpaint", {
    method: "POST",
    body: form,
  });
}

export interface SelectionPoint { x: number; y: number; label: 0 | 1 }
export interface Expansion { left: number; top: number; right: number; bottom: number }

export async function outpaintImage(image: File, parameters: EditParameters & Expansion & { specialized_outpaint?: boolean }): Promise<JobResponse> {
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

/** Selects everything matching a short description, such as "the hat". */
export async function selectByText(image: File, text: string): Promise<JobResponse> {
  const form = new FormData();
  form.set("image", image);
  form.set("parameters", JSON.stringify({ text }));
  return request("/api/select-text", { method: "POST", body: form });
}

/** Selects the main subject with soft, matted edges; `text` names what to keep instead. */
export async function cutoutSubject(image: File, text = ""): Promise<JobResponse> {
  const form = new FormData();
  form.set("image", image);
  form.set("parameters", JSON.stringify({ text }));
  return request("/api/cutout", { method: "POST", body: form });
}

/** Left, top, right and bottom in picture pixels. */
export type FaceBox = [number, number, number, number];

export interface FaceCandidate {
  box: FaceBox;
  score: number;
  /** Inside a bigger person: a real face, which Fix faces picks by itself. */
  suggested: boolean;
}

/** Every face the detector finds in the picture, largest first. */
export async function detectFaces(image: File): Promise<FaceCandidate[]> {
  const form = new FormData();
  form.set("image", image);
  return (await request<{ faces: FaceCandidate[] }>("/api/faces/detect", { method: "POST", body: form })).faces;
}

/** Sharpens the faces, faithfully or by redrawing them; picked `faces`, or else a selection, limit which. */
export async function fixFaces(image: File, parameters: EditParameters & { faithful?: boolean; faces?: FaceBox[] }, selection: Blob | null): Promise<JobResponse> {
  const form = new FormData();
  form.set("image", image);
  form.set("parameters", JSON.stringify(parameters));
  if (selection) form.set("mask", selection, "mask.png");
  return request("/api/faces", { method: "POST", body: form });
}

/** How much detail, from 0.1 to 1: the detail-enhancer LoRA's scale on Quality; Fast redraws more instead. References are not used. */
export type EnhanceParameters = Omit<EditParameters, "reference_ids" | "consistency_lora"> & { detail: number };

/** Adds fine detail to the picture, or only to the selection. */
export async function enhanceImage(image: File, parameters: EnhanceParameters, selection: Blob | null): Promise<JobResponse> {
  const form = new FormData();
  form.set("image", image);
  form.set("parameters", JSON.stringify(parameters));
  if (selection) form.set("mask", selection, "mask.png");
  return request("/api/enhance", { method: "POST", body: form });
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
  layerOffset: { x: number; y: number } = { x: 0, y: 0 },
): Promise<RevisionRecord> {
  return request(`/api/v2/documents/${documentId}/alternatives`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ generation_ids: generationIds, operation_type: operationType, base_revision_id: baseRevisionId, layer_offset_x: layerOffset.x, layer_offset_y: layerOffset.y }),
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

/** A document's references, or the start screen's loose ones for `null`. */
export async function listReferences(documentId: number | null): Promise<ReferenceRecord[]> {
  return request(documentId === null ? "/api/v2/references" : `/api/v2/documents/${documentId}/references`);
}

/** Moves the start screen's references into a new document, so each project keeps its own. */
export async function adoptReferences(documentId: number): Promise<ReferenceRecord[]> {
  return request(`/api/v2/documents/${documentId}/references/adopt`, { method: "POST" });
}

export async function uploadLayerAsset(documentId: number, image: Blob, kind: "image" | "mask", filename = "layer.png"): Promise<LayerAsset> {
  const form = new FormData();
  form.set("image", image, filename);
  form.set("kind", kind);
  return request(`/api/v2/documents/${documentId}/layer-assets`, { method: "POST", body: form });
}

/** Saves the layer stack as an undoable revision; fails with 409 when the document moved on. */
export async function saveLayers(documentId: number, baseRevisionId: number, layers: LayerState[], label: string): Promise<DocumentRecord> {
  const body = layers.map(({ id, name, image_path, mask_path, mask_enabled, variant_paths, x, y, scale_x, scale_y, rotation, opacity, visible, original_id }) =>
    ({ id, name, image_path, mask_path, mask_enabled, variant_paths, x, y, scale_x, scale_y, rotation, opacity, visible, original_id }));
  return request(`/api/v2/documents/${documentId}/layers`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ base_revision_id: baseRevisionId, layers: body, label }),
  });
}

export async function addReference(documentId: number | null, image: File): Promise<ReferenceRecord> {
  const form = new FormData();
  form.set("image", image);
  return request(documentId === null ? "/api/v2/references" : `/api/v2/documents/${documentId}/references`, { method: "POST", body: form });
}

export async function updateReference(referenceId: number, payload: { name?: string; role?: ReferenceRole }): Promise<ReferenceRecord> {
  return request(`/api/v2/references/${referenceId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function deleteReference(referenceId: number): Promise<void> {
  await request(`/api/v2/references/${referenceId}`, { method: "DELETE" });
}

export async function setReferenceIsolation(referenceId: number, isolated: boolean): Promise<ReferenceRecord> {
  return request(`/api/v2/references/${referenceId}/subject`, { method: isolated ? "POST" : "DELETE" });
}

export async function deleteGeneration(id: number): Promise<void> {
  await request(`/api/history/${id}`, { method: "DELETE" });
}

export async function deleteDocument(id: number): Promise<void> {
  await request(`/api/v2/documents/${id}`, { method: "DELETE" });
}

/** Cuts the canvas to a rectangle in picture pixels, without AI; past the picture it becomes transparent. */
export async function cropDocument(documentId: number, baseRevisionId: number, rect: { x: number; y: number; width: number; height: number }): Promise<DocumentRecord> {
  return request(`/api/v2/documents/${documentId}/crop`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ base_revision_id: baseRevisionId, ...rect }),
  });
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
