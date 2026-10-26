import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// In development the backend runs separately (python -m server.main);
// requests to /api are forwarded to it so no CORS setup is needed.
// Point at a backend on another machine with e.g.
//   ISEG_API_URL=http://192.168.1.50:8000 npm run dev
const apiUrl = process.env.ISEG_API_URL ?? 'http://127.0.0.1:8000';

export default defineConfig({
  plugins: [react()],
  // Relative asset URLs, so the build also works under a path such as
  // https://<server>/iseg/ behind a reverse proxy.
  base: './',
  server: {
    proxy: {
      '/api': apiUrl,
    },
  },
});
