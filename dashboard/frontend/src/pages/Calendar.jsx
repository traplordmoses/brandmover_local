import { useState, useEffect } from 'react'
import { ChevronLeft, ChevronRight, Circle, Clock, Tag, Image, ChevronDown, ChevronUp, ExternalLink } from 'lucide-react'
import useApi from '../hooks/useApi'

const STATUS_COLORS = {
  posted: 'var(--success)',
  pending: 'var(--warning)',
  cancelled: 'var(--text-muted)',
  failed: 'var(--error)',
  generating: 'var(--accent)',
  scheduled: 'var(--accent)',
  draft: 'var(--text-secondary)',
}

function formatDate(dateStr) {
  const d = new Date(dateStr + 'T00:00:00')
  return d.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' })
}

function getWeekDates(offset = 0) {
  const now = new Date()
  const day = now.getDay()
  const monday = new Date(now)
  monday.setDate(now.getDate() - (day === 0 ? 6 : day - 1) + offset * 7)

  const dates = []
  for (let i = 0; i < 7; i++) {
    const d = new Date(monday)
    d.setDate(monday.getDate() + i)
    dates.push(d.toISOString().split('T')[0])
  }
  return dates
}

function PostCard({ post }) {
  const [expanded, setExpanded] = useState(false)
  const statusColor = STATUS_COLORS[post.status] || 'var(--text-muted)'
  const isGenerating = post.status === 'generating'

  return (
    <div
      className="rounded-lg border p-3 cursor-pointer transition-colors"
      style={{
        background: 'var(--bg-card)',
        borderColor: 'var(--border)',
      }}
      onMouseEnter={e => e.currentTarget.style.background = 'var(--bg-card-hover)'}
      onMouseLeave={e => e.currentTarget.style.background = 'var(--bg-card)'}
      onClick={() => setExpanded(!expanded)}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="font-mono text-xs" style={{ color: 'var(--text-muted)' }}>
              <Clock size={10} className="inline mr-1" />
              {post.time || '---'}
            </span>
            <span
              className="text-xs px-1.5 py-0.5 rounded-full font-mono"
              style={{
                color: statusColor,
                background: statusColor + '20',
                animation: isGenerating ? 'pulse 2s infinite' : 'none',
              }}
            >
              {post.status}
            </span>
          </div>
          <p className="text-sm truncate" style={{ color: 'var(--text-primary)' }}>
            {post.caption || post.content_type || 'Untitled'}
          </p>
          {post.campaign && (
            <div className="flex items-center gap-1 mt-1">
              <Tag size={10} style={{ color: 'var(--accent)' }} />
              <span className="text-xs" style={{ color: 'var(--accent)' }}>{post.campaign}</span>
            </div>
          )}
        </div>
        <div className="flex items-center gap-1">
          {post.has_image && <Image size={14} style={{ color: 'var(--text-muted)' }} />}
          {expanded ? <ChevronUp size={14} style={{ color: 'var(--text-muted)' }} /> : <ChevronDown size={14} style={{ color: 'var(--text-muted)' }} />}
        </div>
      </div>

      {expanded && (
        <div className="mt-3 pt-3 border-t text-sm" style={{ borderColor: 'var(--border)' }}>
          {post.caption && (
            <p className="mb-2 whitespace-pre-wrap" style={{ color: 'var(--text-secondary)' }}>
              {post.caption}
            </p>
          )}
          {post.content_type && (
            <div className="flex items-center gap-2 mb-1">
              <span className="font-mono text-xs" style={{ color: 'var(--text-muted)' }}>type:</span>
              <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>{post.content_type}</span>
            </div>
          )}
          {post.image_url && (
            <div className="mt-2">
              <img src={post.image_url} alt="" className="rounded-lg max-h-48 object-cover" />
            </div>
          )}
          {post.tweet_url && (
            <a
              href={post.tweet_url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 mt-2 text-xs hover:underline"
              style={{ color: 'var(--accent)' }}
              onClick={e => e.stopPropagation()}
            >
              <ExternalLink size={10} /> View on X
            </a>
          )}
          {post.error && (
            <p className="mt-1 text-xs" style={{ color: 'var(--error)' }}>Error: {post.error}</p>
          )}
        </div>
      )}
    </div>
  )
}

export default function CalendarPage() {
  const [weekOffset, setWeekOffset] = useState(0)
  const weekDates = getWeekDates(weekOffset)
  const startDate = weekDates[0]
  const endDate = weekDates[6]

  const { data, loading } = useApi(`/api/calendar?start=${startDate}&end=${endDate}`, 15000)

  const postsByDate = {}
  weekDates.forEach(d => { postsByDate[d] = [] })
  if (data?.posts) {
    data.posts.forEach(post => {
      const date = post.date || (post.timestamp ? new Date(post.timestamp * 1000).toISOString().split('T')[0] : null)
      if (date && postsByDate[date]) {
        postsByDate[date].push(post)
      }
    })
  }

  const today = new Date().toISOString().split('T')[0]

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-bold" style={{ color: 'var(--text-primary)' }}>Content Calendar</h2>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setWeekOffset(w => w - 1)}
            className="p-2 rounded-lg border cursor-pointer hover:opacity-80 transition-opacity"
            style={{ borderColor: 'var(--border)', background: 'var(--bg-card)', color: 'var(--text-secondary)' }}
          >
            <ChevronLeft size={16} />
          </button>
          <button
            onClick={() => setWeekOffset(0)}
            className="px-3 py-2 rounded-lg border cursor-pointer text-xs font-mono hover:opacity-80 transition-opacity"
            style={{ borderColor: 'var(--border)', background: 'var(--bg-card)', color: 'var(--text-secondary)' }}
          >
            This Week
          </button>
          <button
            onClick={() => setWeekOffset(w => w + 1)}
            className="p-2 rounded-lg border cursor-pointer hover:opacity-80 transition-opacity"
            style={{ borderColor: 'var(--border)', background: 'var(--bg-card)', color: 'var(--text-secondary)' }}
          >
            <ChevronRight size={16} />
          </button>
        </div>
      </div>

      {loading && !data && (
        <div className="text-center py-12 font-mono text-sm" style={{ color: 'var(--text-muted)' }}>
          Loading calendar...
        </div>
      )}

      <div className="grid grid-cols-7 gap-3">
        {weekDates.map(date => {
          const posts = postsByDate[date] || []
          const isToday = date === today
          return (
            <div key={date} className="min-h-48">
              <div
                className="text-xs font-mono mb-2 pb-2 border-b flex items-center gap-2"
                style={{ borderColor: 'var(--border)', color: isToday ? 'var(--accent)' : 'var(--text-muted)' }}
              >
                {isToday && <Circle size={6} fill="var(--accent)" stroke="var(--accent)" />}
                {formatDate(date)}
              </div>
              <div className="flex flex-col gap-2">
                {posts.length === 0 && (
                  <div className="text-xs font-mono py-4 text-center" style={{ color: 'var(--text-muted)' }}>
                    ---
                  </div>
                )}
                {posts.map((post, i) => (
                  <PostCard key={post.id || i} post={post} />
                ))}
              </div>
            </div>
          )
        })}
      </div>

      {data?.stats && (
        <div className="mt-6 flex items-center gap-6 text-xs font-mono" style={{ color: 'var(--text-muted)' }}>
          <span>total: {data.stats.total || 0}</span>
          <span>posted: {data.stats.posted || 0}</span>
          <span>pending: {data.stats.pending || 0}</span>
          <span>failed: {data.stats.failed || 0}</span>
        </div>
      )}
    </div>
  )
}
