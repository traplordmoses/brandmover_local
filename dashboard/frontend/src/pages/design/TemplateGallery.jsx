import { useState } from 'react'
import { LayoutTemplate, Check, Search } from 'lucide-react'

function TemplateCard({ template, isSelected, onSelect }) {
  const name = template.name || template.id || 'Untitled'
  const ratio = template.aspect_ratio || null
  const types = template.content_types || []

  return (
    <button
      onClick={() => onSelect(template)}
      style={{
        display: 'flex',
        flexDirection: 'column',
        padding: '14px',
        borderRadius: '14px',
        background: isSelected ? 'var(--accent-glow)' : 'var(--bg-card)',
        border: isSelected ? '2px solid var(--accent)' : '1px solid var(--border)',
        cursor: 'pointer',
        textAlign: 'left',
        transition: 'all 0.15s',
        minHeight: '120px',
        position: 'relative',
      }}
      onMouseEnter={e => {
        if (!isSelected) e.currentTarget.style.background = 'var(--bg-card-hover)'
      }}
      onMouseLeave={e => {
        if (!isSelected) e.currentTarget.style.background = 'var(--bg-card)'
      }}
    >
      {/* Selected indicator */}
      {isSelected && (
        <div style={{
          position: 'absolute',
          top: '10px',
          right: '10px',
          width: '22px',
          height: '22px',
          borderRadius: '50%',
          background: 'var(--accent)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}>
          <Check size={12} style={{ color: '#fff' }} />
        </div>
      )}

      {/* Template icon area */}
      <div style={{
        width: '40px',
        height: '40px',
        borderRadius: '10px',
        background: 'var(--bg-secondary)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: '10px',
      }}>
        <LayoutTemplate size={18} style={{ color: isSelected ? 'var(--accent)' : 'var(--text-muted)' }} />
      </div>

      {/* Name */}
      <div style={{
        fontSize: '13px',
        fontWeight: 600,
        color: 'var(--text-primary)',
        marginBottom: '6px',
        lineHeight: 1.3,
      }}>
        {name}
      </div>

      {/* Badges */}
      <div style={{
        display: 'flex',
        flexWrap: 'wrap',
        gap: '4px',
        marginTop: 'auto',
      }}>
        {ratio && (
          <span style={{
            fontSize: '10px',
            fontFamily: 'var(--mono)',
            padding: '3px 8px',
            borderRadius: '8px',
            background: 'var(--accent-glow)',
            color: 'var(--accent)',
            fontWeight: 600,
          }}>
            {ratio}
          </span>
        )}
        {types.slice(0, 3).map((ct, i) => (
          <span key={i} style={{
            fontSize: '10px',
            padding: '3px 8px',
            borderRadius: '8px',
            background: 'var(--bg-secondary)',
            color: 'var(--text-muted)',
          }}>
            {typeof ct === 'string' ? ct : ct.id || ct.name || ''}
          </span>
        ))}
        {types.length > 3 && (
          <span style={{
            fontSize: '10px',
            padding: '3px 8px',
            borderRadius: '8px',
            background: 'var(--bg-secondary)',
            color: 'var(--text-muted)',
          }}>
            +{types.length - 3}
          </span>
        )}
      </div>
    </button>
  )
}

export default function TemplateGallery({ templates, onSelect }) {
  const [selectedId, setSelectedId] = useState(null)
  const [search, setSearch] = useState('')

  const handleSelect = (template) => {
    setSelectedId(template.id)
    onSelect(template)
  }

  const filtered = search.trim()
    ? templates.filter(t => {
        const s = search.toLowerCase()
        const name = (t.name || t.id || '').toLowerCase()
        const types = (t.content_types || []).map(ct =>
          (typeof ct === 'string' ? ct : ct.id || ct.name || '').toLowerCase()
        )
        return name.includes(s) || types.some(ct => ct.includes(s))
      })
    : templates

  return (
    <div style={{ padding: '16px' }}>
      {/* Search */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        padding: '10px 14px',
        borderRadius: '12px',
        background: 'var(--bg-card)',
        border: '1px solid var(--border)',
        marginBottom: '16px',
      }}>
        <Search size={16} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />
        <input
          type="text"
          value={search}
          onChange={e => setSearch(e.target.value)}
          placeholder="Search templates..."
          style={{
            flex: 1,
            background: 'none',
            border: 'none',
            outline: 'none',
            color: 'var(--text-primary)',
            fontSize: '13px',
            fontFamily: 'inherit',
            minHeight: '24px',
          }}
        />
      </div>

      {/* Count */}
      <div style={{
        fontSize: '11px',
        color: 'var(--text-muted)',
        fontFamily: 'var(--mono)',
        marginBottom: '12px',
      }}>
        {filtered.length} template{filtered.length !== 1 ? 's' : ''}
      </div>

      {/* Grid */}
      {filtered.length > 0 ? (
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(2, 1fr)',
          gap: '10px',
        }}>
          {filtered.map(template => (
            <TemplateCard
              key={template.id || template.name}
              template={template}
              isSelected={selectedId === template.id}
              onSelect={handleSelect}
            />
          ))}
        </div>
      ) : (
        <div style={{
          textAlign: 'center',
          padding: '48px 16px',
          color: 'var(--text-muted)',
          fontSize: '13px',
        }}>
          {templates.length === 0
            ? 'No templates available'
            : 'No templates match your search'
          }
        </div>
      )}
    </div>
  )
}
