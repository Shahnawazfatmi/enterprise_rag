/*
 * Embeddable chat widget for other websites:
 *
 *   <script src="https://rag.consiva.tech/widget.js" defer></script>
 *
 * Renders the same App as rag.consiva.tech inside a shadow root, so the
 * host page's CSS cannot change the chat and the chat's CSS cannot touch
 * the page. Every deploy of rag.consiva.tech reaches every site at once.
 *
 * Optional attribute on the script tag:
 *   data-api-url="https://apirag.consiva.tech/ask"   (default: VITE_API_URL)
 */

import { createRoot } from 'react-dom/client'
import App from './App.jsx'
import appStyles from './App.css?inline'

const HOST_ID = 'consiva-chat-widget'

// The page's inherited styles (font, colour, line height) stop at the host;
// these replace what index.css gives the chat on rag.consiva.tech.
const BASE_STYLES = `
:host { all: initial; }
.consiva-widget { font-family: Arial, sans-serif; font-size: 16px; line-height: normal; }
*, *::before, *::after { box-sizing: border-box; }
`

// Read while the script runs: document.currentScript is null afterwards.
const script = document.currentScript

// A relative URL resolves against the script's own origin, not the host page.
const apiUrl = new URL(
  script?.dataset.apiUrl || import.meta.env.VITE_API_URL || '/api/ask',
  script?.src || window.location.href,
).href

function mount() {
  // The script may be included twice (layout + page); mount only once.
  if (document.getElementById(HOST_ID)) return

  const host = document.createElement('div')
  host.id = HOST_ID
  document.body.appendChild(host)

  const shadow = host.attachShadow({ mode: 'open' })
  const style = document.createElement('style')
  style.textContent = BASE_STYLES + appStyles

  const container = document.createElement('div')
  container.className = 'consiva-widget'
  shadow.append(style, container)

  createRoot(container).render(<App apiUrl={apiUrl} />)
}

if (document.body) {
  mount()
} else {
  document.addEventListener('DOMContentLoaded', mount)
}
