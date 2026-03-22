import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:5000',
        changeOrigin: true,
        // Timeout 10 phút – cần thiết khi backend xử lý chunk lớn (ML inference chậm)
        timeout: 600_000,
        proxyTimeout: 600_000,
      },
      '/health': 'http://127.0.0.1:5000',
    }
  }
})
