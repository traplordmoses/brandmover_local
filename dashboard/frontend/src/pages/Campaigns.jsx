import { useState } from 'react'
import { ChevronRight, Circle, Calendar, ArrowLeft } from 'lucide-react'
import useApi from '../hooks/useApi'

const STATUS_COLORS = {
  active: 'var(--success)',
  completed: 'var(--accent)',
  draft: 'var(--text-muted)',
  paused: 'var(--warning)',
  cancelled: 'var(--error)',
}

const SLOT_STATUS_COLORS = {
  posted: 'var(--success)',
  pending: 'var(--warning)',
  scheduled: 'var(--accent)',
  generating: 'var(--accent)',
  failed: 'var(--error)',
  cancelled: 'var(--text-muted)',
  draft: 'var(--text-secondary)',
}

function formatDate(dateStr) {
  if (!dateStr) return '---'
  return new Date(dateStr).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
}

function CampaignDetail({ campaign, onBack }) {
  const { data, loading } = useApi(`/api/campaigns/${encodeURIComponent(campaign.id || campaign.name)}`)
  const detail = data || campaign

  const slots = detail.slots || detail.posts || []
  const posted = slots.filter(s => s.status === 'posted').length
  const total = slots.length

  return (
    <div>
      <button
        onClick={onBack}
        className="flex items-center gap-2 mb-4 text-sm cursor-pointer hover:opacity-80 transition-opacity"
        style={{ color: 'var(--accent)', background: 'none', border: 'none' }}
      >
        <ArrowLeft size={14} /> Back to campaigns
      </button>

      <div className="rounded-lg border p-6 mb-6" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}>
        <div className="flex items-start justify-between mb-4">
          <div>
            <h3 className="text-lg font-bold" style={{ color: 'var(--text-primary)' }}>{detail.name}</h3>
            {detail.description && (
              <p className="text-sm mt-1" style={{ color: 'var(--text-secondary)' }}>{detail.description}</p>
            )}
          </div>
          <span
            className="text-xs px-2 py-1 rounded-full font-mono"
            style={{
              color: STATUS_COLORS[detail.status] || 'var(--text-muted)',
              background: (STATUS_COLORS[detail.status] || 'var(--text-muted)') + '20',
            }}
          >
            {detail.status || 'unknown'}
          </span>
        </div>
        <div className="flex items-center gap-6 text-xs font-mono" style={{ color: 'var(--text-muted)' }}>
          <span><Calendar size={10} className="inline mr-1" />{formatDate(detail.start_date)} - {formatDate(detail.end_date)}</span>
          <span>{posted}/{total} posted</span>
        </div>
        {total > 0 && (
          <div className="mt-3 h-1.5 rounded-full overflow-hidden" style={{ background: 'var(--bg-secondary)' }}>
            <div
              className="h-full rounded-full transition-all"
              style={{ width: `${(posted / total) * 100}%`, background: 'var(--success)' }}
            />
          </div>
        )}
      </div>

      <h4 className="text-sm font-mono mb-3" style={{ color: 'var(--text-muted)' }}>Timeline ({total} slots)</h4>

      {loading && slots.length === 0 && (
        <div className="text-center py-8 font-mono text-xs" style={{ color: 'var(--text-muted)' }}>Loading slots...</div>
      )}

      <div className="space-y-2">
        {slots.map((slot, i) => {
          const slotColor = SLOT_STATUS_COLORS[slot.status] || 'var(--text-muted)'
          return (
            <div
              key={slot.id || i}
              className="rounded-lg border p-4 flex items-center gap-4"
              style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}
            >
              <div className="flex-shrink-0 w-1 h-8 rounded-full" style={{ background: slotColor }} />
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-3 mb-1">
                  <span className="font-mono text-xs" style={{ color: 'var(--text-muted)' }}>
                    {slot.date ? formatDate(slot.date) : slot.timestamp ? new Date(slot.timestamp * 1000).toLocaleString() : `Slot ${i + 1}`}
                  </span>
                  <span className="font-mono text-xs" style={{ color: 'var(--text-muted)' }}>
                    {slot.time || ''}
                  </span>
                  <span
                    className="text-xs px-1.5 py-0.5 rounded-full font-mono"
                    style={{ color: slotColor, background: slotColor + '20' }}
                  >
                    {slot.status || 'pending'}
                  </span>
                  {slot.content_type && (
                    <span className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>{slot.content_type}</span>
                  )}
                </div>
                <p className="text-sm truncate" style={{ color: 'var(--text-primary)' }}>
                  {slot.caption || slot.content || slot.label || '---'}
                </p>
              </div>
              {slot.image_url && (
                <img src={slot.image_url} alt="" className="w-12 h-12 rounded object-cover flex-shrink-0" />
              )}
            </div>
          )
        })}
        {slots.length === 0 && !loading && (
          <div className="text-center py-8 font-mono text-xs" style={{ color: 'var(--text-muted)' }}>No slots in this campaign</div>
        )}
      </div>
    </div>
  )
}

export default function CampaignsPage() {
  const { data, loading } = useApi('/api/campaigns', 30000)
  const [selectedCampaign, setSelectedCampaign] = useState(null)

  if (selectedCampaign) {
    return <CampaignDetail campaign={selectedCampaign} onBack={() => setSelectedCampaign(null)} />
  }

  const campaigns = data?.campaigns || []

  return (
    <div>
      <h2 className="text-xl font-bold mb-6" style={{ color: 'var(--text-primary)' }}>Campaigns</h2>

      {loading && campaigns.length === 0 && (
        <div className="text-center py-12 font-mono text-sm" style={{ color: 'var(--text-muted)' }}>Loading campaigns...</div>
      )}

      {!loading && campaigns.length === 0 && (
        <div className="text-center py-12">
          <p className="font-mono text-sm" style={{ color: 'var(--text-muted)' }}>No campaigns found</p>
          <p className="text-xs mt-2" style={{ color: 'var(--text-muted)' }}>Create campaigns through the Telegram bot</p>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {campaigns.map((campaign, i) => {
          const statusColor = STATUS_COLORS[campaign.status] || 'var(--text-muted)'
          const slots = campaign.slots || campaign.posts || []
          const posted = slots.filter(s => s.status === 'posted').length
          const total = campaign.total_posts || slots.length || 0
          const delivered = campaign.delivered || posted
          const progress = total > 0 ? (delivered / total) * 100 : 0

          return (
            <div
              key={campaign.id || campaign.name || i}
              className="rounded-lg border p-5 cursor-pointer transition-colors"
              style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}
              onMouseEnter={e => e.currentTarget.style.background = 'var(--bg-card-hover)'}
              onMouseLeave={e => e.currentTarget.style.background = 'var(--bg-card)'}
              onClick={() => setSelectedCampaign(campaign)}
            >
              <div className="flex items-start justify-between mb-3">
                <div className="flex-1 min-w-0">
                  <h3 className="font-bold text-sm truncate" style={{ color: 'var(--text-primary)' }}>
                    {campaign.name}
                  </h3>
                  {campaign.description && (
                    <p className="text-xs mt-1 truncate" style={{ color: 'var(--text-secondary)' }}>{campaign.description}</p>
                  )}
                </div>
                <div className="flex items-center gap-2 flex-shrink-0 ml-3">
                  <span
                    className="text-xs px-2 py-0.5 rounded-full font-mono"
                    style={{ color: statusColor, background: statusColor + '20' }}
                  >
                    {campaign.status || 'draft'}
                  </span>
                  <ChevronRight size={14} style={{ color: 'var(--text-muted)' }} />
                </div>
              </div>

              <div className="flex items-center gap-4 mb-3 text-xs font-mono" style={{ color: 'var(--text-muted)' }}>
                <span>{formatDate(campaign.start_date)} - {formatDate(campaign.end_date)}</span>
              </div>

              <div className="flex items-center gap-3">
                <div className="flex-1 h-1.5 rounded-full overflow-hidden" style={{ background: 'var(--bg-secondary)' }}>
                  <div
                    className="h-full rounded-full transition-all"
                    style={{ width: `${progress}%`, background: statusColor }}
                  />
                </div>
                <span className="text-xs font-mono flex-shrink-0" style={{ color: 'var(--text-muted)' }}>
                  {delivered}/{total}
                </span>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
