import path from 'path'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    proxy: {
      '/api/ws': {
        target: 'ws://localhost:8000',
        ws: true,
      },
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          'react-vendor': ['react', 'react-dom'],
          'deckgl-vendor': ['@deck.gl/core', '@deck.gl/layers', '@deck.gl/react'],
          'three-vendor': ['three', '@react-three/fiber', '@react-three/drei'],
          'query-vendor': ['@tanstack/react-query', 'zustand'],
        },
      },
    },
  },
})
