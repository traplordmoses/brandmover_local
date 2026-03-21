import { useState, useRef, useEffect, useCallback } from 'react'
import { Send, Loader, Zap, User, Bot, RefreshCw } from 'lucide-react'

function tryParseSpec(text) {
  // Look for ```json blocks in the message
  const jsonMatch = text.match(/```json\s*\n?([\s\S]*?)```/)
  if (jsonMatch) {
    try {
      return JSON.parse(jsonMatch[1].trim())
    } catch {
      return null
    }
  }
  return null
}

function ChatMessage({ message, onUseSpec }) {
  const isUser = message.role === 'user'
  const spec = !isUser ? tryParseSpec(message.content) : null

  // Render message content, stripping the JSON block if a spec was parsed
  const renderContent = () => {
    let text = message.content
    if (spec) {
      text = text.replace(/```json\s*\n?[\s\S]*?```/, '').trim()
    }
    return text
  }

  const displayText = renderContent()

  return (
    <div style={{
      display: 'flex',
      justifyContent: isUser ? 'flex-end' : 'flex-start',
      padding: '4px 0',
    }}>
      <div style={{
        display: 'flex',
        gap: '8px',
        maxWidth: '85%',
        flexDirection: isUser ? 'row-reverse' : 'row',
        alignItems: 'flex-start',
      }}>
        {/* Avatar */}
        <div style={{
          width: '28px',
          height: '28px',
          borderRadius: '50%',
          background: isUser ? 'var(--accent)' : 'var(--bg-card)',
          border: isUser ? 'none' : '1px solid var(--border)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          flexShrink: 0,
        }}>
          {isUser
            ? <User size={14} style={{ color: '#fff' }} />
            : <Bot size={14} style={{ color: 'var(--accent)' }} />
          }
        </div>

        {/* Bubble */}
        <div>
          <div style={{
            padding: '10px 14px',
            borderRadius: isUser ? '16px 16px 4px 16px' : '16px 16px 16px 4px',
            background: isUser ? 'var(--accent)' : 'var(--bg-card)',
            color: isUser ? '#fff' : 'var(--text-primary)',
            fontSize: '13px',
            lineHeight: 1.5,
            border: isUser ? 'none' : '1px solid var(--border)',
            wordBreak: 'break-word',
            whiteSpace: 'pre-wrap',
          }}>
            {displayText}
          </div>

          {/* Spec Card */}
          {spec && (
            <div style={{
              marginTop: '8px',
              background: 'var(--bg-card)',
              border: '1px solid var(--accent)',
              borderRadius: '12px',
              padding: '12px',
            }}>
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                marginBottom: '8px',
              }}>
                <Zap size={14} style={{ color: 'var(--accent)' }} />
                <span style={{
                  fontSize: '12px',
                  fontWeight: 600,
                  color: 'var(--accent)',
                }}>
                  Design Spec Ready
                </span>
              </div>

              <div style={{
                fontSize: '11px',
                fontFamily: 'var(--mono)',
                color: 'var(--text-secondary)',
                background: 'var(--bg-secondary)',
                borderRadius: '8px',
                padding: '8px 10px',
                maxHeight: '120px',
                overflow: 'auto',
                marginBottom: '10px',
                lineHeight: 1.4,
              }}>
                {JSON.stringify(spec, null, 2)}
              </div>

              <button
                onClick={() => onUseSpec(spec)}
                style={{
                  width: '100%',
                  padding: '10px',
                  borderRadius: '10px',
                  background: 'var(--accent)',
                  color: '#fff',
                  border: 'none',
                  fontSize: '13px',
                  fontWeight: 600,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '6px',
                  transition: 'opacity 0.15s',
                  minHeight: '44px',
                }}
                onMouseEnter={e => e.currentTarget.style.opacity = '0.85'}
                onMouseLeave={e => e.currentTarget.style.opacity = '1'}
              >
                <Zap size={14} />
                Use This Spec
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function TypingIndicator() {
  return (
    <div style={{
      display: 'flex',
      alignItems: 'flex-start',
      gap: '8px',
      padding: '4px 0',
    }}>
      <div style={{
        width: '28px',
        height: '28px',
        borderRadius: '50%',
        background: 'var(--bg-card)',
        border: '1px solid var(--border)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexShrink: 0,
      }}>
        <Bot size={14} style={{ color: 'var(--accent)' }} />
      </div>
      <div style={{
        padding: '12px 16px',
        borderRadius: '16px 16px 16px 4px',
        background: 'var(--bg-card)',
        border: '1px solid var(--border)',
        display: 'flex',
        gap: '4px',
        alignItems: 'center',
      }}>
        <span style={{ fontSize: '11px', color: 'var(--text-muted)', marginRight: '4px' }}>Thinking</span>
        {[0, 1, 2].map(i => (
          <span
            key={i}
            style={{
              width: '6px',
              height: '6px',
              borderRadius: '50%',
              background: 'var(--text-muted)',
              animation: `bounce 1.4s infinite ${i * 0.2}s`,
            }}
          />
        ))}
        <style>{`
          @keyframes bounce {
            0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
            40% { transform: translateY(-6px); opacity: 1; }
          }
        `}</style>
      </div>
    </div>
  )
}

const QUICK_PROMPTS = [
  'Create a bold announcement post',
  'Design a product showcase',
  'Build a community engagement post',
  'Make a behind-the-scenes story',
]

export default function DesignChat({ brandData, sessionUploads, onSpecReady, api }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const [error, setError] = useState(null)
  const scrollRef = useRef(null)
  const inputRef = useRef(null)

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages, sending])

  const sendMessage = useCallback(async (text) => {
    if (!text.trim() || sending) return

    const userMsg = { role: 'user', content: text.trim() }
    const newMessages = [...messages, userMsg]
    setMessages(newMessages)
    setInput('')
    setSending(true)
    setError(null)

    const body = {
      messages: newMessages.map(m => ({ role: m.role, content: m.content })),
      session_uploads: sessionUploads.length > 0 ? sessionUploads : null,
    }

    const result = await api.post('/chat', body)

    if (result && result.response) {
      setMessages(prev => [...prev, { role: 'assistant', content: result.response }])
    } else {
      setError(api.error || 'Failed to get a response')
    }

    setSending(false)

    // Refocus input on mobile
    setTimeout(() => inputRef.current?.focus(), 100)
  }, [messages, sending, brandData, sessionUploads, api])

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage(input)
    }
  }

  const handleQuickPrompt = (prompt) => {
    sendMessage(prompt)
  }

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
    }}>
      {/* Messages */}
      <div
        ref={scrollRef}
        style={{
          flex: 1,
          overflow: 'auto',
          padding: '16px',
          display: 'flex',
          flexDirection: 'column',
          gap: '8px',
        }}
      >
        {messages.length === 0 && (
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            flex: 1,
            gap: '16px',
            padding: '24px 16px',
          }}>
            <div style={{
              width: '48px',
              height: '48px',
              borderRadius: '50%',
              background: 'var(--accent-glow)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}>
              <Bot size={24} style={{ color: 'var(--accent)' }} />
            </div>
            <div style={{ textAlign: 'center' }}>
              <div style={{
                fontSize: '15px',
                fontWeight: 600,
                color: 'var(--text-primary)',
                marginBottom: '4px',
              }}>
                Design Agent
              </div>
              <div style={{
                fontSize: '12px',
                color: 'var(--text-muted)',
                maxWidth: '260px',
                lineHeight: 1.5,
              }}>
                Describe what you want to create. I will build a design spec tailored to your brand.
              </div>
            </div>

            {/* Quick Prompts */}
            <div style={{
              display: 'flex',
              flexDirection: 'column',
              gap: '8px',
              width: '100%',
              maxWidth: '320px',
              marginTop: '8px',
            }}>
              {QUICK_PROMPTS.map((prompt, i) => (
                <button
                  key={i}
                  onClick={() => handleQuickPrompt(prompt)}
                  style={{
                    padding: '12px 14px',
                    borderRadius: '12px',
                    background: 'var(--bg-card)',
                    border: '1px solid var(--border)',
                    color: 'var(--text-secondary)',
                    fontSize: '13px',
                    textAlign: 'left',
                    cursor: 'pointer',
                    transition: 'background 0.15s, border-color 0.15s',
                    minHeight: '44px',
                    display: 'flex',
                    alignItems: 'center',
                  }}
                  onMouseEnter={e => {
                    e.currentTarget.style.background = 'var(--bg-card-hover)'
                    e.currentTarget.style.borderColor = 'var(--accent)'
                  }}
                  onMouseLeave={e => {
                    e.currentTarget.style.background = 'var(--bg-card)'
                    e.currentTarget.style.borderColor = 'var(--border)'
                  }}
                >
                  {prompt}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <ChatMessage
            key={i}
            message={msg}
            onUseSpec={onSpecReady}
          />
        ))}

        {sending && <TypingIndicator />}

        {error && !sending && (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '8px',
            padding: '10px',
          }}>
            <span style={{ fontSize: '12px', color: 'var(--error)' }}>{error}</span>
            <button
              onClick={() => {
                setError(null)
                // Retry last user message
                const lastUser = [...messages].reverse().find(m => m.role === 'user')
                if (lastUser) {
                  // Remove last user message to re-send
                  setMessages(prev => prev.slice(0, -1))
                  sendMessage(lastUser.content)
                }
              }}
              style={{
                padding: '4px 10px',
                borderRadius: '8px',
                background: 'var(--bg-card)',
                border: '1px solid var(--border)',
                color: 'var(--text-secondary)',
                fontSize: '11px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
                minHeight: '32px',
              }}
            >
              <RefreshCw size={10} /> Retry
            </button>
          </div>
        )}
      </div>

      {/* Input */}
      <div style={{
        borderTop: '1px solid var(--border)',
        padding: '10px 12px',
        background: 'var(--bg-secondary)',
        flexShrink: 0,
      }}>
        <div style={{
          display: 'flex',
          gap: '8px',
          alignItems: 'flex-end',
        }}>
          <textarea
            ref={inputRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Describe your design..."
            rows={1}
            style={{
              flex: 1,
              padding: '12px 14px',
              borderRadius: '14px',
              background: 'var(--bg-card)',
              border: '1px solid var(--border)',
              color: 'var(--text-primary)',
              fontSize: '14px',
              resize: 'none',
              outline: 'none',
              fontFamily: 'inherit',
              lineHeight: 1.4,
              maxHeight: '100px',
              minHeight: '44px',
              transition: 'border-color 0.15s',
            }}
            onFocus={e => e.currentTarget.style.borderColor = 'var(--accent)'}
            onBlur={e => e.currentTarget.style.borderColor = 'var(--border)'}
            onInput={e => {
              // Auto-resize
              e.currentTarget.style.height = 'auto'
              e.currentTarget.style.height = Math.min(e.currentTarget.scrollHeight, 100) + 'px'
            }}
          />
          <button
            onClick={() => sendMessage(input)}
            disabled={!input.trim() || sending}
            style={{
              width: '44px',
              height: '44px',
              borderRadius: '50%',
              background: input.trim() && !sending ? 'var(--accent)' : 'var(--bg-card)',
              border: 'none',
              color: input.trim() && !sending ? '#fff' : 'var(--text-muted)',
              cursor: input.trim() && !sending ? 'pointer' : 'default',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              flexShrink: 0,
              transition: 'background 0.15s, color 0.15s',
            }}
          >
            {sending
              ? <Loader size={18} style={{ animation: 'spin 1s linear infinite' }} />
              : <Send size={18} />
            }
          </button>
        </div>
        <style>{`@keyframes spin { from { transform: rotate(0deg) } to { transform: rotate(360deg) } }`}</style>
      </div>
    </div>
  )
}
