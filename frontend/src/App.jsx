import ReactMarkdown from 'react-markdown'
import { useEffect, useRef, useState } from 'react'
import './App.css'

/*
 * Backend endpoint. Set VITE_API_URL in frontend/.env for a remote API,
 * e.g. VITE_API_URL=http://172.16.10.117:8033/ask
 * Default '/api/ask' goes through the Vite dev proxy or the nginx proxy.
 */
const API_URL = import.meta.env.VITE_API_URL || '/api/ask'

const REQUEST_TIMEOUT_MS = 60000

// Earlier turns sent with each question so follow-ups make sense.
const HISTORY_MESSAGES = 6

const newId = () =>
  globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`

const ERROR_MESSAGES = {
  429: 'You are sending questions too quickly. Please wait a minute and try again.',
  503: 'The assistant is temporarily unavailable. Please try again in a little while.',
}

async function askBackend(question, history) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS)

  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, history }),
      signal: controller.signal,
    })

    if (!response.ok) {
      throw new Error(
        ERROR_MESSAGES[response.status] ||
          'Sorry, something went wrong. Please try again.',
      )
    }

    return await response.json()
  } catch (error) {
    if (error.name === 'AbortError') {
      throw new Error('The assistant took too long to answer. Please try again.')
    }
    if (error instanceof TypeError) {
      throw new Error('Sorry, I could not connect to the assistant right now. Please try again.')
    }
    throw error
  } finally {
    clearTimeout(timer)
  }
}

function SourceList({ sources }) {
  if (!sources?.length) return null

  return (
    <div className="message-sources">
      <span className="sources-label">Sources</span>
      <ol>
        {sources.map((source) => (
          <li key={source.id} value={source.id}>
            {source.url ? (
              <a href={source.url} target="_blank" rel="noreferrer">
                {source.title}
              </a>
            ) : (
              source.title
            )}
          </li>
        ))}
      </ol>
    </div>
  )
}

/*
 * Inline SVG logo.
 * No image file or external asset required.
 */
function DiamondLogo() {
  return (
    <svg
      viewBox="0 0 64 64"
      xmlns="http://www.w3.org/2000/svg"
      role="img"
      aria-label="Consiva"
      className="consiva-diamond-logo"
    >
      <defs>
        <linearGradient id="diamondTop" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#67e8f9" />
          <stop offset="100%" stopColor="#0284c7" />
        </linearGradient>
        <linearGradient id="diamondLeft" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#bae6fd" />
          <stop offset="100%" stopColor="#38bdf8" />
        </linearGradient>
        <linearGradient id="diamondRight" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#2563eb" />
          <stop offset="100%" stopColor="#1d4ed8" />
        </linearGradient>
      </defs>

      <path d="M13 10H25L32 17L39 10H51L61 24L32 59L3 24Z" fill="#0284c7" />
      <path d="M13 10H25L32 24H3Z" fill="url(#diamondTop)" />
      <path d="M25 10H39L32 24Z" fill="#e0f2fe" />
      <path d="M39 10H51L61 24H32Z" fill="#38bdf8" />
      <path d="M3 24H32L22 42Z" fill="url(#diamondLeft)" />
      <path d="M32 24H61L42 42Z" fill="url(#diamondRight)" />
      <path d="M22 42L32 24L42 42L32 59Z" fill="#1e40af" />
      <path d="M3 24H32L22 42Z" fill="#7dd3fc" />
      <path d="M32 24H61L42 42Z" fill="#2563eb" />
      <path d="M22 42L32 24L32 59Z" fill="#38bdf8" />
      <path d="M32 24L42 42L32 59Z" fill="#1d4ed8" />
    </svg>
  )
}

function App() {
  const [isOpen, setIsOpen] = useState(false)
  const [input, setInput] = useState('')
  const [messages, setMessages] = useState([])
  const [isThinking, setIsThinking] = useState(false)
  const messagesEndRef = useRef(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({
      behavior: 'smooth',
    })
  }, [messages, isThinking])

  const sendMessage = async () => {
    const question = input.trim()

    if (!question || isThinking) return

    const userMessage = {
      id: newId(),
      role: 'user',
      content: question,
    }

    // Successful turns only; error notices are not conversation.
    const history = messages
      .filter((message) => !message.isError)
      .slice(-HISTORY_MESSAGES)
      .map(({ role, content }) => ({ role, content }))

    setMessages((previous) => [...previous, userMessage])
    setInput('')
    setIsThinking(true)

    try {
      const data = await askBackend(question, history)

      const assistantMessage = {
        id: newId(),
        role: 'assistant',
        content:
          data.answer ||
          'I could not find an answer to that question.',
        sources: data.sources || [],
      }

      setMessages((previous) => [...previous, assistantMessage])
    } catch (error) {
      console.error('Chat API error:', error)

      const errorMessage = {
        id: newId(),
        role: 'assistant',
        content: error.message,
        isError: true,
      }

      setMessages((previous) => [...previous, errorMessage])
    } finally {
      setIsThinking(false)
    }
  }

  const handleKeyDown = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      sendMessage()
    }
  }

  const toggleChat = () => {
    setIsOpen((previous) => !previous)
  }

  return (
    <>
      {isOpen && (
        <div className="consiva-chat">
          <div className="chat-glow" />

          <header className="chat-header">
            <div className="assistant-info">
              <div className="assistant-avatar">
                <DiamondLogo />
              </div>

              <div className="assistant-details">
                <h3>AI Assistant</h3>

                <div className="online-status">
                  <span className="online-dot" />
                  <span>Online</span>
                </div>
              </div>
            </div>

            <button
              type="button"
              className="header-close"
              onClick={() => setIsOpen(false)}
              aria-label="Close chat"
            >
              ×
            </button>
          </header>

          <main className="chat-body">
            {messages.length === 0 && (
              <div className="welcome-section">
                <div className="welcome-label">
                  AI ASSISTANT
                </div>

                <div className="bot-message">
                  <p>
                    👋 Hi! How can I help you with Consiva today?
                  </p>
                </div>
              </div>
            )}

            {messages.map((message) => (
              <div
                key={message.id}
                className={`message-row ${message.role}`}
              >
                <div
                  className={`message-bubble ${message.role}`}
                >
                  {message.role === 'assistant' ? (
                    <>
                      <ReactMarkdown>
                        {message.content}
                      </ReactMarkdown>
                      <SourceList sources={message.sources} />
                    </>
                  ) : (
                    message.content
                  )}
                </div>
              </div>
            ))}

            {isThinking && (
              <div className="message-row assistant">
                <div className="thinking-bubble">
                  <span className="thinking-text">
                    Thinking
                  </span>
                  <span className="thinking-dot" />
                  <span className="thinking-dot" />
                  <span className="thinking-dot" />
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </main>

          <div className="input-area">
            <div className="chat-input-wrapper">
              <textarea
                value={input}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask anything related to Consiva AI..."
                rows={1}
                disabled={isThinking}
                aria-label="Ask AI Assistant"
              />

              <button
                type="button"
                className="send-button"
                onClick={sendMessage}
                disabled={!input.trim() || isThinking}
                aria-label="Send message"
              >
                <span>→</span>
              </button>
            </div>

            <div className="input-hint">
              Press Enter to send
            </div>
          </div>

          <div className="powered-by">
            AI Assistant
          </div>
        </div>
      )}

      <button
        type="button"
        className="chat-launcher"
        onClick={toggleChat}
        aria-label={
          isOpen ? 'Close AI Assistant' : 'Open AI Assistant'
        }
      >
        <span className="launcher-icon">
          <DiamondLogo />
        </span>
      </button>
    </>
  )
}

export default App
