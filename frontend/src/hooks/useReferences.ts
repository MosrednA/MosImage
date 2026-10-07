import { useEffect, useRef, useState } from "react";
import { ReferenceRecord, addReference, deleteReference, listReferences, setReferenceIsolation } from "../api";

const ACCEPTED = new Set(["image/png", "image/jpeg", "image/webp"]);

/** Concept images of the open document, plus the one used for the next edit. */
export function useReferences(documentId: number | null, onError: (message: string) => void) {
  const [references, setReferences] = useState<ReferenceRecord[]>([]);
  const [activeId, setActiveId] = useState<number | null>(null);
  const [pending, setPending] = useState<ReadonlySet<number>>(new Set());
  const [uploading, setUploading] = useState(0);
  const current = useRef(documentId);
  current.current = documentId;

  useEffect(() => {
    setReferences([]); setActiveId(null); setPending(new Set()); setUploading(0);
    if (documentId === null) return;
    let disposed = false;
    listReferences(documentId)
      .then(items => { if (!disposed) setReferences(items); })
      .catch(err => { if (!disposed) onError(message(err)); });
    return () => { disposed = true; };
  }, [documentId]);

  /** Uploads the images and returns the one to use right away, if any. */
  async function add(files: File[]): Promise<number | null> {
    const documentAtStart = documentId;
    if (documentAtStart === null) return null;
    const images = files.filter(file => ACCEPTED.has(file.type));
    if (images.length < files.length) onError("References can be PNG, JPG or WebP images.");
    if (!images.length) return null;
    setUploading(count => count + images.length);
    let firstAdded: number | null = null;
    for (const file of images) {
      try {
        const record = await addReference(documentAtStart, file);
        if (current.current !== documentAtStart) return null;
        setReferences(items => [...items, record]);
        firstAdded ??= record.id;
      } catch (err) {
        onError(message(err));
      } finally {
        if (current.current === documentAtStart) setUploading(count => count - 1);
      }
    }
    // Adding a single reference is a clear intent to use it right away.
    return images.length === 1 ? firstAdded : null;
  }

  async function remove(id: number) {
    await withPending(id, async () => {
      await deleteReference(id);
      setReferences(items => items.filter(item => item.id !== id));
      setActiveId(active => active === id ? null : active);
    });
  }

  async function setIsolated(id: number, isolated: boolean) {
    await withPending(id, async () => {
      const record = await setReferenceIsolation(id, isolated);
      setReferences(items => items.map(item => item.id === id ? record : item));
    });
  }

  async function withPending(id: number, action: () => Promise<void>) {
    setPending(ids => new Set(ids).add(id));
    try { await action(); } catch (err) { onError(message(err)); }
    finally { setPending(ids => { const next = new Set(ids); next.delete(id); return next; }); }
  }

  const active = references.find(item => item.id === activeId) ?? null;
  return { references, active, setActiveId, pending, uploading, add, remove, setIsolated };
}

function message(error: unknown): string { return error instanceof Error ? error.message : "The reference could not be updated."; }
