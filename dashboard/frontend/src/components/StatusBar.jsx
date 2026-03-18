import { useState, useEffect } from 'react'
import { Circle, Pause, Play } from 'lucide-react'

function relativeTime(ts) {
  if (!ts) return 'never'
  const diff = Date.now() / 1000 - ts
  if (diff < 0) return 'just now'
  if (diff < 60) return `${Math.floor(diff)}s ago`
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`
  return `${Math.floor(diff / 86400)}d ago`
}

export default function StatusBar() {
  const [status, setStatus] = useState(null)

  useEffect(() => {
    const load = () => fetch('/api/status').then(r => r.json()).then(setStatus).catch(() => {})
    load()
    const id = setInterval(load, 10000)
    return () => clearInterval(id)
  }, [])

  const togglePause = async () => {
    await fetch('/api/status/pause', { method: 'POST' })
    const r = await fetch('/api/status')
    setStatus(await r.json())
  }

  if (!status) return null

  const isRunning = !status.paused
  const statusColor = isRunning ? 'var(--success)' : 'var(--warning)'

  return (
    <div className="h-10 flex items-center justify-between px-4 border-b text-xs font-mono" style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border)', color: 'var(--text-muted)' }}>
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          <Circle size={8} fill={statusColor} stroke={statusColor} />
          <span style={{ color: statusColor }}>{isRunning ? 'RUNNING' : 'PAUSED'}</span>
        </div>
        <span>heartbeat: {relativeTime(status.last_heartbeat)}</span>
        <span>last post: {relativeTime(status.last_post_timestamp)}</span>
        <span>mode: {status.agent_mode || 'unknown'}</span>
      </div>
      <div className="flex items-center gap-3">
        {status.next_scheduled && (
          <span>next: {status.next_scheduled.label} @ {new Date(status.next_scheduled.timestamp * 1000).toLocaleTimeString()}</span>
        )}
        <button
          onClick={togglePause}
          className="flex items-center gap-1 px-2 py-1 rounded border transition-colors cursor-pointer hover:opacity-80"
          style={{ borderColor: 'var(--border)', color: statusColor, background: 'transparent' }}
        >
          {isRunning ? <><Pause size={12} /> pause</> : <><Play size={12} /> resume</>}
        </button>
      </div>
    </div>
  )
}
