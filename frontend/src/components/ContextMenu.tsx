import { KeyboardEvent, useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

export interface MenuItem {
  label: string;
  shortcut?: string;
  title?: string;
  disabled?: boolean;
  danger?: boolean;
  onSelect: () => void;
}
export type MenuEntry = MenuItem | "separator";

interface Props {
  /** Where it opens, in window pixels; it moves in to stay on screen. */
  x: number;
  y: number;
  label: string;
  items: MenuEntry[];
  onClose: () => void;
}

/** A right-click menu: arrow keys move, Enter picks, Esc or a click elsewhere closes it. */
export function ContextMenu({ x, y, label, items, onClose }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const close = useRef(onClose);
  close.current = onClose;
  const [place, setPlace] = useState({ left: x, top: y });

  useLayoutEffect(() => {
    const menu = ref.current;
    if (!menu) return;
    const { width, height } = menu.getBoundingClientRect();
    setPlace({ left: Math.max(4, Math.min(x, window.innerWidth - width - 4)), top: Math.max(4, Math.min(y, window.innerHeight - height - 4)) });
    menu.querySelector<HTMLButtonElement>("button:not(:disabled)")?.focus();
  }, [x, y]);

  useEffect(() => {
    const away = (event: Event) => { if (!ref.current?.contains(event.target as Node)) close.current(); };
    const dismiss = () => close.current();
    window.addEventListener("pointerdown", away, true);
    window.addEventListener("wheel", away, true);
    window.addEventListener("blur", dismiss);
    window.addEventListener("resize", dismiss);
    return () => {
      window.removeEventListener("pointerdown", away, true);
      window.removeEventListener("wheel", away, true);
      window.removeEventListener("blur", dismiss);
      window.removeEventListener("resize", dismiss);
    };
  }, []);

  function key(event: KeyboardEvent<HTMLDivElement>) {
    // The canvas shortcuts wait until the menu is gone.
    event.stopPropagation();
    if (event.key === "Escape" || event.key === "Tab") { event.preventDefault(); onClose(); return; }
    if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
    event.preventDefault();
    const buttons = Array.from(ref.current?.querySelectorAll<HTMLButtonElement>("button:not(:disabled)") ?? []);
    const index = buttons.indexOf(document.activeElement as HTMLButtonElement);
    buttons[(index + (event.key === "ArrowDown" ? 1 : -1) + buttons.length) % buttons.length]?.focus();
  }

  return createPortal(<div ref={ref} className="context-menu" role="menu" aria-label={label} style={place} onKeyDown={key} onContextMenu={event => event.preventDefault()}>
    {items.map((item, index) => item === "separator"
      ? <div key={"separator-" + index} className="context-menu-separator" role="separator" />
      : <button key={item.label} type="button" role="menuitem" disabled={item.disabled} title={item.title} className={item.danger ? "danger" : undefined}
        onClick={() => { onClose(); item.onSelect(); }}>
        <span>{item.label}</span>{item.shortcut && <kbd>{item.shortcut}</kbd>}
      </button>)}
  </div>, document.body);
}
