import { useState, useEffect } from 'react'
import { Clock, ChevronDown, ChevronUp, ExternalLink, Image, RefreshCw, AlertCircle } from 'lucide-react'

function formatTime(ts) {
  if (!ts) return '---'
  const d = typeof ts === 'number' ? new Date(ts * 1000) : new Date(ts)
  return d.toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function relativeTime(ts) {
  if (!ts) return ''
  const now = Date.now()
  const then = typeof ts === 'number' ? ts * 1000 : new Date(ts).getTime()
  const diff = Math.floor((now - then) / 1000)
  if (diff < 0) return 'just now'
  if (diff < 60) return `${diff}s ago`
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`
  return `${Math.floor(diff / 86400)}d ago`
}

function HistoryCard({ entry }) {
  const [expanded, setExpanded] = useState(false)

  const contentType = entry.content_type || entry.type || null
  const caption = entry.caption || entry.content || null
  const imageUrl = entry.image_url || null
  const timestamp = entry.timestamp || entry.created_at || null
  const status = entry.status || null

  const statusColor = status === 'posted' ? 'var(--success)'
    : status === 'failed' ? 'var(--error)'
    : status === 'draft' ? 'var(--warning)'
    : 'var(--text-muted)'

  return (
    <div
      onClick={() => setExpanded(!expanded)}
      style={{
        background: 'var(--bg-card)',
        border: '1px solid var(--border)',
        borderRadius: '12px',
        padding: '14px',
        cursor: 'pointer',
        transition: 'background 0.15s',
      }}
      onMouseEnter={e => e.currentTarget.style.background = 'var(--bg-card-hover)'}
      onMouseLeave={e => e.currentTarget.style.background = 'var(--bg-card)'}
    >
      {/* Header Row */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
      }}>
        {/* Timestamp */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '4px',
          fontSize: '11px',
          color: 'var(--text-muted)',
          fontFamily: 'var(--mono)',
          flexShrink: 0,
        }}>
          <Clock size={10} />
          {relativeTime(timestamp)}
        </div>

        {/* Content Type Badge */}
        {contentType && (
          <span style={{
            fontSize: '10px',
            padding: '3px 8px',
            borderRadius: '8px',
            background: 'var(--accent-glow)',
            color: 'var(--accent)',
            fontWeight: 600,
            flexShrink: 0,
          }}>
            {contentType}
          </span>
        )}

        {/* Status Badge */}
        {status && (
          <span style={{
            fontSize: '10px',
            padding: '3px 8px',
            borderRadius: '8px',
            background: statusColor + '15',
            color: statusColor,
            fontFamily: 'var(--mono)',
            flexShrink: 0,
          }}>
            {status}
          </span>
        )}

        {/* Image indicator */}
        {imageUrl && (
          <Image size={12} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />
        )}

        {/* Spacer */}
        <div style={{ flex: 1 }} />

        {/* Expand toggle */}
        {expanded
          ? <ChevronUp size={14} style={{ color: 'var(--text-muted)' }} />
          : <ChevronDown size={14} style={{ color: 'var(--text-muted)' }} />
        }
      </div>

      {/* Caption Preview */}
      {caption && (
        <p style={{
          fontSize: '12px',
          color: 'var(--text-secondary)',
          marginTop: '8px',
          lineHeight: 1.4,
          overflow: 'hidden',
          display: '-webkit-box',
          WebkitLineClamp: expanded ? 'unset' : 2,
          WebkitBoxOrient: 'vertical',
          whiteSpace: expanded ? 'pre-wrap' : undefined,
        }}>
          {caption}
        </p>
      )}

      {/* Expanded Details */}
      {expanded && (
        <div style={{
          marginTop: '12px',
          paddingTop: '12px',
          borderTop: '1px solid var(--border)',
        }}>
          {/* Full timestamp */}
          <div style={{
            fontSize: '11px',
            color: 'var(--text-muted)',
            fontFamily: 'var(--mono)',
            marginBottom: '8px',
          }}>
            {formatTime(timestamp)}
          </div>

          {/* Image preview */}
          {imageUrl && (
            <img
              src={imageUrl}
              alt=""
              style={{
                width: '100%',
                maxHeight: '240px',
                objectFit: 'cover',
                borderRadius: '10px',
                marginBottom: '10px',
              }}
              onClick={e => e.stopPropagation()}
            />
          )}

          {/* Spec details */}
          {entry.spec && (
            <details style={{ marginBottom: '8px' }}>
              <summary style={{
                fontSize: '11px',
                color: 'var(--text-muted)',
                cursor: 'pointer',
                fontFamily: 'var(--mono)',
                padding: '4px 0',
              }}>
                Design spec
              </summary>
              <pre style={{
                fontSize: '10px',
                fontFamily: 'var(--mono)',
                color: 'var(--text-secondary)',
                background: 'var(--bg-secondary)',
                padding: '8px',
                borderRadius: '8px',
                overflow: 'auto',
                maxHeight: '150px',
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
                marginTop: '4px',
              }}
                onClick={e => e.stopPropagation()}
              >
                {JSON.stringify(entry.spec, null, 2)}
              </pre>
            </details>
          )}

          {/* External link */}
          {entry.tweet_url && (
            <a
              href={entry.tweet_url}
              target="_blank"
              rel="noopener noreferrer"
              onClick={e => e.stopPropagation()}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '4px',
                fontSize: '11px',
                color: 'var(--accent)',
                textDecoration: 'none',
              }}
            >
              <ExternalLink size={10} /> View on X
            </a>
          )}

          {entry.error && (
            <p style={{ fontSize: '11px', color: 'var(--error)', marginTop: '6px' }}>
              {entry.error}
            </p>
          )}
        </div>
      )}
    </div>
  )
}

export default function HistoryView({ api }) {
  const [entries, setEntries] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const loadHistory = async () => {
    setLoading(true)
    setError(null)
    const data = await api.get('/history')
    if (data) {
      setEntries(data.entries || data.history || [])
    } else {
      setError(api.error || 'Failed to load history')
    }
    setLoading(false)
  }

  useEffect(() => {
    loadHistory()
  }, [])

  return (
    <div style={{ padding: '16px' }}>
      {/* Header */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginBottom: '16px',
      }}>
        <span style={{
          fontSize: '11px',
          color: 'var(--text-muted)',
          fontFamily: 'var(--mono)',
        }}>
          {entries.length} generation{entries.length !== 1 ? 's' : ''}
        </span>
        <button
          onClick={loadHistory}
          disabled={loading}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            padding: '6px 12px',
            borderRadius: '8px',
            background: 'var(--bg-card)',
            border: '1px solid var(--border)',
            color: 'var(--text-secondary)',
            fontSize: '11px',
            fontFamily: 'var(--mono)',
            cursor: loading ? 'wait' : 'pointer',
            minHeight: '32px',
            transition: 'background 0.15s',
          }}
          onMouseEnter={e => e.currentTarget.style.background = 'var(--bg-card-hover)'}
          onMouseLeave={e => e.currentTarget.style.background = 'var(--bg-card)'}
        >
          <RefreshCw size={10} style={loading ? { animation: 'spin 1s linear infinite' } : {}} />
          Refresh
        </button>
      </div>

      {/* Loading */}
      {loading && entries.length === 0 && (
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '48px 16px',
          gap: '12px',
        }}>
          <RefreshCw size={20} style={{ color: 'var(--text-muted)', animation: 'spin 1s linear infinite' }} />
          <span style={{ fontSize: '12px', color: 'var(--text-muted)', fontFamily: 'var(--mono)' }}>
            Loading history...
          </span>
        </div>
      )}

      {/* Error */}
      {error && !loading && (
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '12px',
          padding: '32px 16px',
        }}>
          <AlertCircle size={24} style={{ color: 'var(--error)' }} />
          <span style={{ fontSize: '12px', color: 'var(--error)' }}>{error}</span>
          <button
            onClick={loadHistory}
            style={{
              padding: '8px 16px',
              borderRadius: '10px',
              background: 'var(--bg-card)',
              border: '1px solid var(--border)',
              color: 'var(--text-secondary)',
              fontSize: '12px',
              cursor: 'pointer',
              minHeight: '36px',
            }}
          >
            Retry
          </button>
        </div>
      )}

      {/* Empty */}
      {!loading && !error && entries.length === 0 && (
        <div style={{
          textAlign: 'center',
          padding: '48px 16px',
          color: 'var(--text-muted)',
          fontSize: '13px',
        }}>
          No generations yet. Use the Composer or Agent to create your first design.
        </div>
      )}

      {/* Entries */}
      {entries.length > 0 && (
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          gap: '8px',
        }}>
          {entries.map((entry, i) => (
            <HistoryCard key={entry.id || i} entry={entry} />
          ))}
        </div>
      )}

      <style>{`@keyframes spin { from { transform: rotate(0deg) } to { transform: rotate(360deg) } }`}</style>
    </div>
  )
}
