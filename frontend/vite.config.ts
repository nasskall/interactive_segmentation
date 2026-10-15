import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// In development the backend runs separately (python -m server.main);
// requests to /api are forwarded to it so no CORS setup is needed.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8000',
    },
  },
});
