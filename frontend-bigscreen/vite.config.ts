import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

function stripGeneratedTrailingWhitespace() {
  return {
    name: "strip-generated-trailing-whitespace",
    enforce: "post" as const,
    renderChunk(code: string) {
      return { code: code.replace(/[ \t]+$/gm, ""), map: null };
    }
  };
}

export default defineConfig({
  base: "/web-v2/",
  plugins: [react(), stripGeneratedTrailingWhitespace()],
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
