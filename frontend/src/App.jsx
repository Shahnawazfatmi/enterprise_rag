import ReactMarkdown from 'react-markdown'
import { useEffect, useRef, useState } from 'react'
import './App.css'

/*
  FastAPI backend endpoint.
  Change this later if your backend is deployed.
*/
const API_URL = 'http://172.16.10.117:8033/ask'
function App() {
  // Controls whether the chatbot window is open
  const [isOpen, setIsOpen] = useState(false)

  // Stores what the user is currently typing
  const [input, setInput] = useState('')

  // Stores the complete conversation
  const [messages, setMessages] = useState([])

  // True while waiting for FastAPI/RAG response
  const [isThinking, setIsThinking] = useState(false)

  // Used to automatically scroll to the newest message
  const messagesEndRef = useRef(null)

  /*
    Automatically scroll to the latest message
    whenever messages or thinking state changes.
  */
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({
      behavior: 'smooth',
    })
  }, [messages, isThinking])

  /*
    Sends the user's question to FastAPI.
  */
  const sendMessage = async () => {
    const question = input.trim()

    // Don't send empty messages
    if (!question || isThinking) return

    // Add user's message immediately to the chat
    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: question,
    }

    setMessages((previous) => [
      ...previous,
      userMessage,
    ])

    // Clear input box
    setInput('')

    // Show "Thinking..." while backend works
    setIsThinking(true)

    try {
      /*
        Send question to FastAPI.
      */
      const response = await fetch(API_URL, {
        method: 'POST',

        headers: {
          'Content-Type': 'application/json',
        },

        body: JSON.stringify({
          question: question,
        }),
      })

      /*
        Handle HTTP errors such as:
        400, 404, 500, etc.
      */
      if (!response.ok) {
        throw new Error(
          `Server error: ${response.status}`
        )
      }

      /*
        Convert FastAPI JSON response
        into a JavaScript object.
      */
      const data = await response.json()

      /*
        Add the actual RAG answer to the chat.
      */
      const assistantMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content:
          data.answer ||
          'I could not find an answer to that question.',
      }

      setMessages((previous) => [
        ...previous,
        assistantMessage,
      ])

    } catch (error) {
      /*
        Show a clean error message
        if the backend is unavailable.
      */
      console.error('Chat API error:', error)

      const errorMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content:
          'Sorry, I could not connect to the assistant right now. Please try again.',
      }

      setMessages((previous) => [
        ...previous,
        errorMessage,
      ])

    } finally {
      /*
        Hide "Thinking..." whether
        request succeeds or fails.
      */
      setIsThinking(false)
    }
  }

  /*
    Enter = send message
    Shift + Enter = new line
  */
  const handleKeyDown = (event) => {
    if (
      event.key === 'Enter' &&
      !event.shiftKey
    ) {
      event.preventDefault()

      sendMessage()
    }
  }

  /*
    Floating button toggles the chatbot.

    Closed → opens
    Open → closes

    The icon itself stays ✦.
    It does NOT turn into X.
  */
  const toggleChat = () => {
    setIsOpen((previous) => !previous)
  }

  return (
    <>
      {/* =================================
          CHAT WINDOW
          ================================= */}

      {isOpen && (
        <div className="consiva-chat">

          {/* Background blue glow */}
          <div className="chat-glow" />

          {/* =================================
              HEADER
              ================================= */}

          <header className="chat-header">

            <div className="assistant-info">

              {/* Assistant icon */}
              <div className="assistant-avatar">
                <span>✦</span>
              </div>

              {/* Assistant name and status */}
              <div className="assistant-details">

                <h3>AI Assistant</h3>

                <div className="online-status">
                  <span className="online-dot" />
                  <span>Online</span>
                </div>

              </div>

            </div>

            {/* Close button inside chat */}
            <button
              type="button"
              className="header-close"
              onClick={() => setIsOpen(false)}
              aria-label="Close chat"
            >
              ×
            </button>

          </header>

          {/* =================================
              CHAT BODY
              ================================= */}

          <main className="chat-body">

            {/* Show welcome message only
                when conversation is empty */}
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

            {/* =================================
                CONVERSATION MESSAGES
                ================================= */}

            {messages.map((message) => (
              <div
                key={message.id}
                className={`message-row ${message.role}`}
              >

               <div
  className={`message-bubble ${message.role}`}
>
  {message.role === 'assistant' ? (
    <ReactMarkdown>
      {message.content}
    </ReactMarkdown>
  ) : (
    message.content
  )}
</div>

              </div>
            ))}

            {/* =================================
                THINKING INDICATOR
                ================================= */}

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

            {/* Invisible element used
                for automatic scrolling */}
            <div ref={messagesEndRef} />

          </main>

          {/* =================================
              INPUT AREA
              ================================= */}

          <div className="input-area">

            <div className="chat-input-wrapper">

              <textarea
                value={input}
                onChange={(event) =>
                  setInput(event.target.value)
                }
                onKeyDown={handleKeyDown}
                placeholder="Ask anything related to consiva Ai..."
                rows={1}
                disabled={isThinking}
                aria-label="Ask AI Assistant"
              />

              <button
                type="button"
                className="send-button"
                onClick={sendMessage}
                disabled={
                  !input.trim() ||
                  isThinking
                }
                aria-label="Send message"
              >
                <span>→</span>
              </button>

            </div>

            <div className="input-hint">
              Press Enter to send
            </div>

          </div>

          {/* =================================
              FOOTER
              ================================= */}

          <div className="powered-by">
            AI Assistant
          </div>

        </div>
      )}

      {/* =================================
          FLOATING CHAT BUTTON
          
          Clicking it toggles:
          closed → open
          open → closed
          
          Icon ALWAYS remains ✦
          ================================= */}

      <button
        type="button"
        className="chat-launcher"
        onClick={toggleChat}
        aria-label={
          isOpen
            ? 'Close AI Assistant'
            : 'Open AI Assistant'
        }
      >
        <span className="launcher-icon">
          ✦

        </span>
      </button>
    </>
  )
}

export default App