import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "/web-v2/",
  plugins: [react()],
  build: {
    outDir: "../src/quant_lab/web/bigscreen_static",
    emptyOutDir: true,
    rolldownOptions: {
      output: {
        codeSplitting: {
          groups: [
            {
              name: "echarts-charts",
              test: /node_modules[\\/]echarts[\\/]lib[\\/]chart[\\/]/,
              priority: 40,
              maxSize: 450 * 1024
            },
            {
              name: "echarts-components",
              test: /node_modules[\\/]echarts[\\/]lib[\\/](component|coord)[\\/]/,
              priority: 35
            },
            {
              name: "zrender",
              test: /node_modules[\\/]zrender[\\/]/,
              priority: 32
            },
            {
              name: "echarts-core",
              test: /node_modules[\\/](echarts|echarts-for-react)[\\/]/,
              priority: 30
            },
            {
              name: "react-vendor",
              test: /node_modules[\\/](react|react-dom|scheduler|@tanstack)[\\/]/,
              priority: 20
            },
            {
              name: "motion",
              test: /node_modules[\\/]framer-motion[\\/]/,
              priority: 10
            },
            {
              name: "vendor",
              test: /node_modules[\\/]/,
              priority: 1
            }
          ]
        }
      }
    }
  },
  server: {
    proxy: {
      "/v1": "http://127.0.0.1:8027",
      "/web-v2": "http://127.0.0.1:8027"
    }
  }
});
