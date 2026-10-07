import { GenerationRecord } from "../api";

type ImageFetcher = (input: RequestInfo | URL) => Promise<Response>;

export async function generationAsFile(
  record: GenerationRecord,
  imageUrl: string,
  fetcher: ImageFetcher = fetch,
): Promise<File> {
  const response = await fetcher(imageUrl);
  if (!response.ok) {
    throw new Error(`Could not open generation ${record.id}: ${response.status}`);
  }
  const blob = await response.blob();
  const type = blob.type || "image/png";
  return new File([blob], `generation-${record.id}.${extensionFor(type)}`, { type });
}

export function firstDroppedImage(files: FileList): File | null {
  return Array.from(files).find((file) => file.type.startsWith("image/")) ?? null;
}

function extensionFor(type: string): string {
  if (type === "image/jpeg") return "jpg";
  if (type === "image/webp") return "webp";
  return "png";
}
