import react from '@vitejs/plugin-react';
import { defineConfig } from 'vitest/config';

const api = process.env.VITE_API_TARGET ?? 'http://127.0.0.1:8040';

export default defineConfig({
  plugins: [react()],
  server: { port: 5174, proxy: { '/api': api } },
  // `npm run build` writes into the Python package, which serves it (lol-live-capture serve).
  build: { outDir: '../src/lol_live_capture/web', emptyOutDir: true },
  test: { environment: 'jsdom', globals: true, setupFiles: ['./src/test-setup.ts'] },
});
