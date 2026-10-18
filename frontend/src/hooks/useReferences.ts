import { useEffect, useRef, useState } from "react";
import { ReferenceRecord, ReferenceRole, addReference, deleteReference, listReferences, setReferenceIsolation, updateReference } from "../api";

const ACCEPTED = new Set(["image/png", "image/jpeg", "image/webp"]);
/** Matches the backend limit; more references dilute each likeness. */
export const MAX_ACTIVE_REFERENCES = 3;

/**
 * Concept images of the open document, or the start screen's loose ones
 * without a document, plus the ones used for the next edit, in order.
 * They load once the engine has `started`, so the start screen asks nothing of it.
 */
export function useReferences(documentId: number | null, started: boolean, onError: (message: string) => void) {
  const [references, setReferences] = useState<ReferenceRecord[]>([]);
  const [activeIds, setActiveIds] = useState<number[]>([]);
  const [pending, setPending] = useState<ReadonlySet<number>>(new Set());
  const [uploading, setUploading] = useState(0);
  const current = useRef(documentId);
  current.current = documentId;

  useEffect(() => {
    setReferences([]); setActiveIds([]); setPending(new Set()); setUploading(0);
    if (!started) return;
    let disposed = false;
    listReferences(documentId)
      .then(items => { if (!disposed) setReferences(items); })
      .catch(err => { if (!disposed) onError(message(err)); });
    return () => { disposed = true; };
  }, [documentId, started]);

  /** Uploads the images and returns the one to use right away, if any. */
  async function add(files: File[]): Promise<number | null> {
    const documentAtStart = documentId;
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
      setActiveIds(ids => ids.filter(active => active !== id));
    });
  }

  async function setIsolated(id: number, isolated: boolean) {
    await withPending(id, async () => {
      const record = await setReferenceIsolation(id, isolated);
      setReferences(items => items.map(item => item.id === id ? record : item));
    });
  }

  async function setRole(id: number, role: ReferenceRole) {
    await withPending(id, async () => {
      const record = await updateReference(id, { role });
      setReferences(items => items.map(item => item.id === id ? record : item));
    });
  }

  async function withPending(id: number, action: () => Promise<void>) {
    setPending(ids => new Set(ids).add(id));
    try { await action(); } catch (err) { onError(message(err)); }
    finally { setPending(ids => { const next = new Set(ids); next.delete(id); return next; }); }
  }

  /** Adds the reference to the next edit, or takes it out. Returns false at the limit. */
  function toggle(id: number): boolean {
    if (activeIds.includes(id)) { setActiveIds(ids => ids.filter(active => active !== id)); return true; }
    if (activeIds.length >= MAX_ACTIVE_REFERENCES) return false;
    setActiveIds(ids => [...ids, id]);
    return true;
  }
  function clear() { setActiveIds([]); }

  const active = activeIds.map(id => references.find(item => item.id === id)).filter((item): item is ReferenceRecord => !!item);
  return { references, active, toggle, clear, pending, uploading, add, remove, setIsolated, setRole };
}

function message(error: unknown): string { return error instanceof Error ? error.message : "The reference could not be updated."; }
