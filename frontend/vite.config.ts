import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// MOSIMAGE_PORT moves the engine off 8000, for example when Windows has reserved
// that port; the Tauri shell reads it at build time too.
const enginePort = process.env.MOSIMAGE_PORT;

export default defineConfig({
  plugins: [react()],
  define: enginePort && !process.env.VITE_API_BASE_URL
    ? { "import.meta.env.VITE_API_BASE_URL": JSON.stringify(`http://127.0.0.1:${enginePort}`) }
    : {},
  server: {
    port: 5173,
    strictPort: true,
  },
});
