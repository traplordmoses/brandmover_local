import { useState, useEffect } from 'react'
import { Save, RefreshCw, AlertCircle, CheckCircle, Clock, DollarSign, Brain, Server } from 'lucide-react'
import useApi from '../hooks/useApi'

export default function SettingsPage() {
  const { data: settings, loading, reload } = useApi('/api/settings')
  const { data: statsData } = useApi('/api/settings/stats')
  const { data: prefsData } = useApi('/api/settings/preferences')
  const [scheduleText, setScheduleText] = useState('')
  const [originalScheduleText, setOriginalScheduleText] = useState('')
  const [saving, setSaving] = useState(false)
  const [saveStatus, setSaveStatus] = useState(null)

  useEffect(() => {
    if (settings?.schedule) {
      const text = JSON.stringify(settings.schedule, null, 2)
      setScheduleText(text)
      setOriginalScheduleText(text)
    }
  }, [settings])

  const hasChanges = scheduleText !== originalScheduleText

  const handleSaveSchedule = async () => {
    setSaving(true)
    setSaveStatus(null)
    try {
      const parsed = JSON.parse(scheduleText)
      const r = await fetch('/api/settings/schedule', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config: parsed }),
      })
      if (r.ok) {
        setOriginalScheduleText(scheduleText)
        setSaveStatus('saved')
        setTimeout(() => setSaveStatus(null), 3000)
        reload()
      } else {
        const err = await r.json().catch(() => ({}))
        setSaveStatus(err.error || 'Save failed')
      }
    } catch (e) {
      setSaveStatus('Invalid JSON: ' + e.message)
    }
    setSaving(false)
  }

  if (loading && !settings) {
    return <div className="text-center py-12 font-mono text-sm" style={{ color: 'var(--text-muted)' }}>Loading settings...</div>
  }

  const stats = statsData || {}
  const env = settings ? {
    agent_mode: settings.agent_mode,
    brand_name: settings.brand_name,
    auto_post_enabled: settings.auto_post_enabled,
    heartbeat_enabled: settings.heartbeat_enabled,
    content_planner_enabled: settings.content_planner_enabled,
    skeleton_library_enabled: settings.skeleton_library_enabled,
    diversity_tracker_enabled: settings.diversity_tracker_enabled,
    publish_platforms: (settings.publish_platforms || []).join(', '),
  } : {}
  const preferences = prefsData?.learned_preferences
    ? prefsData.learned_preferences.split('\n').filter(l => l.trim().startsWith('- ')).map(l => l.trim().slice(2))
    : []

  return (
    <div className="space-y-6 max-w-4xl">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold" style={{ color: 'var(--text-primary)' }}>Settings</h2>
        <button
          onClick={reload}
          className="flex items-center gap-2 px-3 py-2 rounded-lg border cursor-pointer text-xs font-mono hover:opacity-80 transition-opacity"
          style={{ borderColor: 'var(--border)', background: 'var(--bg-card)', color: 'var(--text-secondary)' }}
        >
          <RefreshCw size={12} /> Refresh
        </button>
      </div>

      {/* Schedule Editor */}
      <div className="rounded-lg border overflow-hidden" style={{ borderColor: 'var(--border)', background: 'var(--bg-card)' }}>
        <div className="flex items-center justify-between px-4 py-3 border-b" style={{ borderColor: 'var(--border)', background: 'var(--bg-secondary)' }}>
          <div className="flex items-center gap-2">
            <Clock size={14} style={{ color: 'var(--accent)' }} />
            <span className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>Schedule Configuration</span>
            {hasChanges && (
              <span className="text-xs font-mono px-1.5 py-0.5 rounded" style={{ color: 'var(--warning)', background: 'var(--warning)' + '20' }}>
                unsaved
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            {saveStatus === 'saved' && (
              <span className="flex items-center gap-1 text-xs font-mono" style={{ color: 'var(--success)' }}>
                <CheckCircle size={10} /> saved
              </span>
            )}
            {saveStatus && saveStatus !== 'saved' && (
              <span className="flex items-center gap-1 text-xs font-mono" style={{ color: 'var(--error)' }}>
                <AlertCircle size={10} /> {saveStatus}
              </span>
            )}
            <button
              onClick={handleSaveSchedule}
              disabled={!hasChanges || saving}
              className="flex items-center gap-1 px-3 py-1 rounded-lg border text-xs font-mono cursor-pointer transition-opacity"
              style={{
                borderColor: hasChanges ? 'var(--accent)' : 'var(--border)',
                color: hasChanges ? 'var(--accent)' : 'var(--text-muted)',
                background: 'transparent',
                opacity: hasChanges ? 1 : 0.5,
              }}
            >
              <Save size={12} /> {saving ? 'Saving...' : 'Save'}
            </button>
          </div>
        </div>
        <textarea
          value={scheduleText}
          onChange={e => setScheduleText(e.target.value)}
          className="w-full border-none outline-none p-4 text-sm resize-none"
          style={{
            background: 'transparent',
            color: 'var(--text-primary)',
            fontFamily: 'var(--mono)',
            lineHeight: 1.6,
            minHeight: '200px',
          }}
          spellCheck={false}
        />
      </div>

      {/* Generation Stats */}
      <div className="rounded-lg border overflow-hidden" style={{ borderColor: 'var(--border)', background: 'var(--bg-card)' }}>
        <div className="flex items-center gap-2 px-4 py-3 border-b" style={{ borderColor: 'var(--border)', background: 'var(--bg-secondary)' }}>
          <DollarSign size={14} style={{ color: 'var(--accent)' }} />
          <span className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>Generation Stats</span>
        </div>
        <div className="p-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <StatCard label="Total Generations" value={stats.total ?? '---'} />
            <StatCard label="Total Cost" value={stats.total_cost != null ? `$${stats.total_cost.toFixed(2)}` : '---'} />
            <StatCard label="Total Feedback" value={stats.total_feedback ?? '---'} />
            <StatCard label="Approval Rate" value={stats.approval_rate != null ? `${(stats.approval_rate * 100).toFixed(1)}%` : '---'} />
          </div>
          {stats.by_model && Object.keys(stats.by_model).length > 0 && (
            <div className="mt-4 pt-4 border-t" style={{ borderColor: 'var(--border)' }}>
              <h4 className="text-xs font-mono mb-3" style={{ color: 'var(--text-muted)' }}>Model Distribution</h4>
              <div className="space-y-2">
                {Object.entries(stats.by_model).map(([model, count]) => {
                  const total = Object.values(stats.by_model).reduce((a, b) => a + b, 0)
                  const pct = total > 0 ? (count / total) * 100 : 0
                  return (
                    <div key={model} className="flex items-center gap-3">
                      <span className="text-xs font-mono w-40 truncate" style={{ color: 'var(--text-secondary)' }}>{model}</span>
                      <div className="flex-1 h-1.5 rounded-full overflow-hidden" style={{ background: 'var(--bg-secondary)' }}>
                        <div className="h-full rounded-full" style={{ width: `${pct}%`, background: 'var(--accent)' }} />
                      </div>
                      <span className="text-xs font-mono w-16 text-right" style={{ color: 'var(--text-muted)' }}>{count} ({pct.toFixed(0)}%)</span>
                    </div>
                  )
                })}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Environment */}
      <div className="rounded-lg border overflow-hidden" style={{ borderColor: 'var(--border)', background: 'var(--bg-card)' }}>
        <div className="flex items-center gap-2 px-4 py-3 border-b" style={{ borderColor: 'var(--border)', background: 'var(--bg-secondary)' }}>
          <Server size={14} style={{ color: 'var(--accent)' }} />
          <span className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>Environment</span>
        </div>
        <div className="p-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {Object.entries(env).length === 0 && (
              <span className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>No environment info available</span>
            )}
            {Object.entries(env).map(([key, value]) => (
              <div key={key} className="flex items-center gap-3">
                <span className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>{key}:</span>
                <span className="text-xs font-mono" style={{ color: 'var(--text-secondary)' }}>
                  {typeof value === 'boolean' ? (value ? 'true' : 'false') : String(value)}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Learned Preferences */}
      <div className="rounded-lg border overflow-hidden" style={{ borderColor: 'var(--border)', background: 'var(--bg-card)' }}>
        <div className="flex items-center gap-2 px-4 py-3 border-b" style={{ borderColor: 'var(--border)', background: 'var(--bg-secondary)' }}>
          <Brain size={14} style={{ color: 'var(--accent)' }} />
          <span className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>Learned Preferences</span>
        </div>
        <div className="p-4">
          {preferences.length === 0 && (
            <span className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>No learned preferences yet</span>
          )}
          <div className="space-y-2">
            {preferences.map((pref, i) => (
              <div
                key={i}
                className="rounded border p-3 text-sm"
                style={{ borderColor: 'var(--border)', background: 'var(--bg-secondary)' }}
              >
                {typeof pref === 'string' ? (
                  <p style={{ color: 'var(--text-secondary)' }}>{pref}</p>
                ) : (
                  <div>
                    {pref.summary && <p style={{ color: 'var(--text-secondary)' }}>{pref.summary}</p>}
                    {pref.category && (
                      <span className="text-xs font-mono mt-1 inline-block" style={{ color: 'var(--text-muted)' }}>
                        [{pref.category}]
                      </span>
                    )}
                    {pref.timestamp && (
                      <span className="text-xs font-mono mt-1 ml-2 inline-block" style={{ color: 'var(--text-muted)' }}>
                        {new Date(pref.timestamp * 1000).toLocaleDateString()}
                      </span>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

function StatCard({ label, value }) {
  return (
    <div className="rounded-lg border p-3" style={{ borderColor: 'var(--border)', background: 'var(--bg-secondary)' }}>
      <p className="text-xs font-mono mb-1" style={{ color: 'var(--text-muted)' }}>{label}</p>
      <p className="text-lg font-bold font-mono" style={{ color: 'var(--text-primary)' }}>{value}</p>
    </div>
  )
}
