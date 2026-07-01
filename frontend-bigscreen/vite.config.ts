import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "/web-v2/",
  plugins: [react()],
  build: {
    outDir: "../src/quant_lab/web/bigscreen_static",
    emptyOutDir: true,
    chunkSizeWarningLimit: 1200
  },
  server: {
    proxy: {
      "/v1": "http://127.0.0.1:8027",
      "/web-v2": "http://127.0.0.1:8027"
    }
  }
});
