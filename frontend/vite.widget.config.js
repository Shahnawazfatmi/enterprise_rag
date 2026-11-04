import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The embeddable chat (src/widget.jsx) as one self-contained script,
// dist/widget.js: React, the chat and its CSS. Built after the site
// (see "build" in package.json), so it must not empty dist/.
export default defineConfig({
  plugins: [react()],
  define: {
    'process.env.NODE_ENV': JSON.stringify('production'),
  },
  build: {
    emptyOutDir: false,
    lib: {
      entry: 'src/widget.jsx',
      name: 'ConsivaChat',
      formats: ['iife'],
      fileName: () => 'widget.js',
      // App.jsx's own CSS import is emitted here, unused: the widget
      // injects the same CSS into its shadow root.
      cssFileName: 'widget',
    },
  },
})
