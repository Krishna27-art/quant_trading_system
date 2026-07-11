import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'node:path'

// Dev server proxies /api and /ws straight to the FastAPI/Flask backend so
// the frontend never hardcodes a host. Point VITE_API_TARGET at wherever
// the backend actually runs (default assumes local dev on :8000).
const target = process.env.VITE_API_TARGET || 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': { target, changeOrigin: true },
      '/ws': { target: target.replace('http', 'ws'), ws: true, changeOrigin: true },
    },
  },
  build: {
    target: 'es2020',
    sourcemap: true,
    chunkSizeWarningLimit: 800,
  },
})
