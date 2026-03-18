import { useState } from 'react'
import { ExternalLink, RefreshCw, Circle } from 'lucide-react'
import useApi from '../hooks/useApi'

function relativeTime(ts) {
  if (!ts) return '---'
  const diff = Date.now() / 1000 - ts
  if (diff < 0) return 'just now'
  if (diff < 60) return `${Math.floor(diff)}s ago`
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`
  return `${Math.floor(diff / 86400)}d ago`
}

function formatTimestamp(ts) {
  if (!ts) return '---'
  return new Date(ts * 1000).toLocaleString('en-US', {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit'
  })
}

const DECISION_COLORS = {
  post: 'var(--success)',
  skip: 'var(--text-muted)',
  defer: 'var(--warning)',
  error: 'var(--error)',
}

export default function StatusPage() {
  const [tab, setTab] = useState('heartbeat')
  const { data: heartbeatData, loading: hbLoading, reload: reloadHb } = useApi('/api/status/heartbeat-log', 10000)
  const { data: activityData, loading: actLoading, reload: reloadAct } = useApi('/api/status/activity', 15000)

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-bold" style={{ color: 'var(--text-primary)' }}>System Status</h2>
        <button
          onClick={() => { reloadHb(); reloadAct() }}
          className="flex items-center gap-2 px-3 py-2 rounded-lg border cursor-pointer text-xs font-mono hover:opacity-80 transition-opacity"
          style={{ borderColor: 'var(--border)', background: 'var(--bg-card)', color: 'var(--text-secondary)' }}
        >
          <RefreshCw size={12} /> Refresh
        </button>
      </div>

      <div className="flex gap-1 mb-4">
        {['heartbeat', 'activity'].map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className="px-3 py-1.5 rounded-lg text-xs font-mono cursor-pointer transition-colors"
            style={{
              background: tab === t ? 'var(--accent-glow)' : 'transparent',
              color: tab === t ? 'var(--accent)' : 'var(--text-muted)',
              border: 'none',
            }}
          >
            {t}
          </button>
        ))}
      </div>

      {tab === 'heartbeat' && (
        <div className="rounded-lg border overflow-hidden" style={{ borderColor: 'var(--border)', background: 'var(--bg-card)' }}>
          <table className="w-full text-sm">
            <thead>
              <tr style={{ background: 'var(--bg-secondary)' }}>
                <th className="text-left px-4 py-2 font-mono text-xs font-normal" style={{ color: 'var(--text-muted)' }}>Time</th>
                <th className="text-left px-4 py-2 font-mono text-xs font-normal" style={{ color: 'var(--text-muted)' }}>Signals</th>
                <th className="text-left px-4 py-2 font-mono text-xs font-normal" style={{ color: 'var(--text-muted)' }}>Decision</th>
                <th className="text-left px-4 py-2 font-mono text-xs font-normal" style={{ color: 'var(--text-muted)' }}>Action</th>
              </tr>
            </thead>
            <tbody>
              {hbLoading && !heartbeatData && (
                <tr><td colSpan={4} className="px-4 py-8 text-center font-mono text-xs" style={{ color: 'var(--text-muted)' }}>Loading...</td></tr>
              )}
              {heartbeatData?.entries?.length === 0 && (
                <tr><td colSpan={4} className="px-4 py-8 text-center font-mono text-xs" style={{ color: 'var(--text-muted)' }}>No heartbeat entries</td></tr>
              )}
              {(heartbeatData?.entries || []).map((entry, i) => {
                const decColor = DECISION_COLORS[entry.decision] || 'var(--text-secondary)'
                return (
                  <tr key={i} className="border-t" style={{ borderColor: 'var(--border)' }}>
                    <td className="px-4 py-2 font-mono text-xs whitespace-nowrap" style={{ color: 'var(--text-muted)' }}>
                      {formatTimestamp(entry.timestamp)}
                    </td>
                    <td className="px-4 py-2 text-xs" style={{ color: 'var(--text-secondary)' }}>
                      {Array.isArray(entry.signals)
                        ? entry.signals.map((s, j) => (
                          <span key={j} className="inline-block mr-1 mb-1 px-1.5 py-0.5 rounded font-mono" style={{ background: 'var(--bg-secondary)', color: 'var(--text-secondary)' }}>
                            {typeof s === 'string' ? s : s.label || JSON.stringify(s)}
                          </span>
                        ))
                        : (entry.signals || '---')
                      }
                    </td>
                    <td className="px-4 py-2 font-mono text-xs" style={{ color: decColor }}>
                      <Circle size={6} fill={decColor} stroke={decColor} className="inline mr-1" />
                      {entry.decision || '---'}
                    </td>
                    <td className="px-4 py-2 text-xs max-w-xs truncate" style={{ color: 'var(--text-secondary)' }}>
                      {entry.action_taken || '---'}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}

      {tab === 'activity' && (
        <div className="space-y-2">
          {actLoading && !activityData && (
            <div className="text-center py-8 font-mono text-xs" style={{ color: 'var(--text-muted)' }}>Loading...</div>
          )}
          {activityData?.entries?.length === 0 && (
            <div className="text-center py-8 font-mono text-xs" style={{ color: 'var(--text-muted)' }}>No activity yet</div>
          )}
          {(activityData?.entries || []).map((entry, i) => (
            <div
              key={i}
              className="rounded-lg border p-4 flex items-start gap-4"
              style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}
            >
              <div className="flex-shrink-0 mt-1">
                <Circle
                  size={8}
                  fill={entry.status === 'posted' ? 'var(--success)' : entry.status === 'failed' ? 'var(--error)' : 'var(--text-muted)'}
                  stroke={entry.status === 'posted' ? 'var(--success)' : entry.status === 'failed' ? 'var(--error)' : 'var(--text-muted)'}
                />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-3 mb-1">
                  <span className="font-mono text-xs" style={{ color: 'var(--text-muted)' }}>
                    {formatTimestamp(entry.timestamp)}
                  </span>
                  <span
                    className="text-xs px-1.5 py-0.5 rounded-full font-mono"
                    style={{
                      color: entry.status === 'posted' ? 'var(--success)' : entry.status === 'failed' ? 'var(--error)' : 'var(--text-muted)',
                      background: (entry.status === 'posted' ? 'var(--success)' : entry.status === 'failed' ? 'var(--error)' : 'var(--text-muted)') + '20',
                    }}
                  >
                    {entry.status || 'unknown'}
                  </span>
                  {entry.content_type && (
                    <span className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>{entry.content_type}</span>
                  )}
                </div>
                <p className="text-sm truncate" style={{ color: 'var(--text-primary)' }}>
                  {entry.caption || entry.content || 'No caption'}
                </p>
                {entry.tweet_url && (
                  <a
                    href={entry.tweet_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1 mt-1 text-xs hover:underline"
                    style={{ color: 'var(--accent)' }}
                  >
                    <ExternalLink size={10} /> View on X
                  </a>
                )}
                {entry.error && (
                  <p className="mt-1 text-xs" style={{ color: 'var(--error)' }}>{entry.error}</p>
                )}
              </div>
              {entry.image_url && (
                <img src={entry.image_url} alt="" className="w-16 h-16 rounded-lg object-cover flex-shrink-0" />
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
