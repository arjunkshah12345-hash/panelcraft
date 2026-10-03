import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The API runs on 8000 locally; proxying keeps the browser same-origin in dev so the
// relative /api paths in the client work identically in dev and production.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.PANELCRAFT_API ?? "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  build: { outDir: "dist", sourcemap: true },
});
