import { Icon } from "./Icon";

export function WelcomeCanvas({ disabled, onOpen }: { disabled: boolean; onOpen: () => void }) {
  return <div className="welcome-canvas">
    <div className="welcome-art" aria-hidden="true"><div /><div /><div /><Icon name="spark" /></div>
    <span className="eyebrow">Your next idea starts here</span>
    <h1>Make something<br /><em>beautiful.</em></h1>
    <p>A new image from a few words.<br />Or give your own image a fresh direction.</p>
    <button disabled={disabled} onClick={onOpen}><Icon name="upload" />Open an image</button>
    <small>Drop an image here · or paste with Ctrl + V</small>
  </div>;
}
