import { useState, useRef } from 'react'
import { Upload, Check, AlertCircle, Palette, Type, Sparkles, Mic, RefreshCw } from 'lucide-react'

function ColorSwatch({ color, label }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    navigator.clipboard?.writeText(color)
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }

  return (
    <button
      onClick={handleCopy}
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: '8px',
        background: 'none',
        border: 'none',
        cursor: 'pointer',
        padding: '8px',
        minWidth: 0,
      }}
    >
      <div style={{
        width: '56px',
        height: '56px',
        borderRadius: '50%',
        background: color,
        border: '2px solid var(--border)',
        boxShadow: `0 0 12px ${color}30`,
        transition: 'transform 0.15s, box-shadow 0.15s',
      }}
        onMouseEnter={e => {
          e.currentTarget.style.transform = 'scale(1.1)'
          e.currentTarget.style.boxShadow = `0 0 20px ${color}50`
        }}
        onMouseLeave={e => {
          e.currentTarget.style.transform = 'scale(1)'
          e.currentTarget.style.boxShadow = `0 0 12px ${color}30`
        }}
      />
      <span style={{
        fontSize: '10px',
        fontFamily: 'var(--mono)',
        color: copied ? 'var(--success)' : 'var(--text-muted)',
        transition: 'color 0.2s',
      }}>
        {copied ? 'Copied' : (label || color)}
      </span>
    </button>
  )
}

function FontSample({ font }) {
  const name = typeof font === 'string' ? font : font.name || font.family || 'Unknown'
  const role = typeof font === 'string' ? null : font.role || font.usage || null

  return (
    <div style={{
      background: 'var(--bg-card)',
      border: '1px solid var(--border)',
      borderRadius: '12px',
      padding: '16px',
      transition: 'background 0.2s',
    }}
      onMouseEnter={e => e.currentTarget.style.background = 'var(--bg-card-hover)'}
      onMouseLeave={e => e.currentTarget.style.background = 'var(--bg-card)'}
    >
      <div style={{
        fontSize: '20px',
        fontWeight: 600,
        color: 'var(--text-primary)',
        marginBottom: '6px',
        lineHeight: 1.3,
      }}>
        {name}
      </div>
      {role && (
        <span style={{
          fontSize: '11px',
          color: 'var(--text-muted)',
          fontFamily: 'var(--mono)',
          textTransform: 'uppercase',
          letterSpacing: '0.5px',
        }}>
          {role}
        </span>
      )}
    </div>
  )
}

function StyleTag({ tag }) {
  return (
    <span style={{
      display: 'inline-block',
      padding: '6px 14px',
      borderRadius: '20px',
      background: 'var(--accent-glow)',
      color: 'var(--accent)',
      fontSize: '12px',
      fontWeight: 500,
      whiteSpace: 'nowrap',
    }}>
      {tag}
    </span>
  )
}

function VoiceCard({ trait }) {
  const label = typeof trait === 'string' ? trait : trait.label || trait.name || ''
  const detail = typeof trait === 'string' ? null : trait.description || trait.detail || null

  return (
    <div style={{
      background: 'var(--bg-card)',
      border: '1px solid var(--border)',
      borderRadius: '10px',
      padding: '12px 14px',
    }}>
      <div style={{
        fontSize: '13px',
        fontWeight: 600,
        color: 'var(--text-primary)',
        marginBottom: detail ? '4px' : 0,
      }}>
        {label}
      </div>
      {detail && (
        <div style={{
          fontSize: '11px',
          color: 'var(--text-muted)',
          lineHeight: 1.4,
        }}>
          {detail}
        </div>
      )}
    </div>
  )
}

function ReferenceResult({ analysis }) {
  return (
    <div style={{
      background: 'var(--bg-card)',
      border: '1px solid var(--border)',
      borderRadius: '12px',
      padding: '14px',
      marginTop: '10px',
    }}>
      <div style={{
        fontSize: '12px',
        fontWeight: 600,
        color: 'var(--text-primary)',
        marginBottom: '8px',
      }}>
        Reference Analysis
      </div>
      {analysis.colors && analysis.colors.length > 0 && (
        <div style={{ marginBottom: '8px' }}>
          <span style={{ fontSize: '10px', color: 'var(--text-muted)', fontFamily: 'var(--mono)' }}>
            EXTRACTED COLORS
          </span>
          <div style={{ display: 'flex', gap: '6px', marginTop: '4px', flexWrap: 'wrap' }}>
            {analysis.colors.map((c, i) => (
              <div key={i} style={{
                width: '28px',
                height: '28px',
                borderRadius: '6px',
                background: c,
                border: '1px solid var(--border)',
              }} />
            ))}
          </div>
        </div>
      )}
      {analysis.keywords && analysis.keywords.length > 0 && (
        <div style={{ marginBottom: '8px' }}>
          <span style={{ fontSize: '10px', color: 'var(--text-muted)', fontFamily: 'var(--mono)' }}>
            STYLE KEYWORDS
          </span>
          <div style={{ display: 'flex', gap: '4px', marginTop: '4px', flexWrap: 'wrap' }}>
            {analysis.keywords.map((kw, i) => (
              <span key={i} style={{
                fontSize: '10px',
                padding: '3px 8px',
                borderRadius: '12px',
                background: 'var(--bg-secondary)',
                color: 'var(--text-secondary)',
              }}>
                {kw}
              </span>
            ))}
          </div>
        </div>
      )}
      {analysis.alignment_score != null && (
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginTop: '6px' }}>
          <span style={{ fontSize: '10px', color: 'var(--text-muted)', fontFamily: 'var(--mono)' }}>
            BRAND ALIGNMENT
          </span>
          <span style={{
            fontSize: '12px',
            fontWeight: 600,
            color: analysis.alignment_score >= 0.7 ? 'var(--success)' : analysis.alignment_score >= 0.4 ? 'var(--warning)' : 'var(--error)',
          }}>
            {Math.round(analysis.alignment_score * 100)}%
          </span>
        </div>
      )}
      {analysis.notes && (
        <p style={{ fontSize: '11px', color: 'var(--text-secondary)', marginTop: '6px', lineHeight: 1.4 }}>
          {analysis.notes}
        </p>
      )}
    </div>
  )
}

function Section({ icon: Icon, title, children }) {
  return (
    <div style={{ marginBottom: '24px' }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        marginBottom: '12px',
        paddingLeft: '2px',
      }}>
        <Icon size={16} style={{ color: 'var(--accent)' }} />
        <span style={{
          fontSize: '13px',
          fontWeight: 600,
          color: 'var(--text-primary)',
          textTransform: 'uppercase',
          letterSpacing: '0.5px',
        }}>
          {title}
        </span>
      </div>
      {children}
    </div>
  )
}

export default function BrandBoard({ brandData, onReferenceAnalyzed, api }) {
  const [uploading, setUploading] = useState(false)
  const [uploadResult, setUploadResult] = useState(null)
  const [uploadError, setUploadError] = useState(null)
  const fileRef = useRef(null)

  const handleUpload = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return

    setUploading(true)
    setUploadError(null)
    setUploadResult(null)

    const result = await api.upload('/analyze-reference', file)
    if (result) {
      setUploadResult(result)
      onReferenceAnalyzed?.(result)
    } else {
      setUploadError(api.error || 'Upload failed')
    }
    setUploading(false)

    // Reset file input
    if (fileRef.current) fileRef.current.value = ''
  }

  if (!brandData) {
    return (
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '48px 16px',
        gap: '12px',
      }}>
        <RefreshCw size={24} style={{ color: 'var(--text-muted)', animation: 'spin 1s linear infinite' }} />
        <span style={{ fontSize: '13px', color: 'var(--text-muted)', fontFamily: 'var(--mono)' }}>
          Loading brand data...
        </span>
        <style>{`@keyframes spin { from { transform: rotate(0deg) } to { transform: rotate(360deg) } }`}</style>
      </div>
    )
  }

  // colors/fonts can be dicts {role: {hex, name}} or arrays — normalize to arrays
  const rawColors = brandData.colors || {}
  const colors = Array.isArray(rawColors)
    ? rawColors
    : Object.entries(rawColors).map(([role, c]) => ({
        hex: c.hex || c.value || '#000',
        name: c.name || role,
        role,
      }))
  const rawFonts = brandData.fonts || {}
  const fonts = Array.isArray(rawFonts)
    ? rawFonts
    : Object.entries(rawFonts).map(([use, f]) => ({
        family: f.family || 'Unknown',
        name: f.family || 'Unknown',
        weight: f.weight || '',
        role: use,
        usage: use,
      }))
  const styleKeywords = brandData.style_keywords || brandData.keywords || []
  const voiceTraits = brandData.voice_traits || brandData.voice || []

  return (
    <div style={{ padding: '16px' }}>
      {/* Brand Name Header */}
      {brandData.brand_name && (
        <div style={{
          textAlign: 'center',
          marginBottom: '24px',
          paddingBottom: '16px',
          borderBottom: '1px solid var(--border)',
        }}>
          <h2 style={{
            fontSize: '22px',
            fontWeight: 700,
            color: 'var(--text-primary)',
            margin: 0,
          }}>
            {brandData.brand_name}
          </h2>
          {brandData.tagline && (
            <p style={{
              fontSize: '13px',
              color: 'var(--text-muted)',
              marginTop: '4px',
            }}>
              {brandData.tagline}
            </p>
          )}
        </div>
      )}

      {/* Colors */}
      {colors.length > 0 && (
        <Section icon={Palette} title="Colors">
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(80px, 1fr))',
            gap: '4px',
          }}>
            {colors.map((color, i) => {
              const hex = typeof color === 'string' ? color : color.hex || color.value || '#000'
              const label = typeof color === 'string' ? color : color.name || color.label || hex
              return <ColorSwatch key={i} color={hex} label={label} />
            })}
          </div>
        </Section>
      )}

      {/* Fonts */}
      {fonts.length > 0 && (
        <Section icon={Type} title="Typography">
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '8px',
          }}>
            {fonts.map((font, i) => (
              <FontSample key={i} font={font} />
            ))}
          </div>
        </Section>
      )}

      {/* Style Keywords */}
      {styleKeywords.length > 0 && (
        <Section icon={Sparkles} title="Style">
          <div style={{
            display: 'flex',
            flexWrap: 'wrap',
            gap: '8px',
          }}>
            {styleKeywords.map((kw, i) => (
              <StyleTag key={i} tag={typeof kw === 'string' ? kw : kw.label || kw.name || ''} />
            ))}
          </div>
        </Section>
      )}

      {/* Voice Traits */}
      {voiceTraits.length > 0 && (
        <Section icon={Mic} title="Voice">
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))',
            gap: '8px',
          }}>
            {voiceTraits.map((trait, i) => (
              <VoiceCard key={i} trait={trait} />
            ))}
          </div>
        </Section>
      )}

      {/* Upload Reference */}
      <Section icon={Upload} title="Reference Upload">
        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          onChange={handleUpload}
          style={{ display: 'none' }}
          id="ref-upload"
        />
        <label
          htmlFor="ref-upload"
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '10px',
            padding: '20px',
            borderRadius: '12px',
            border: '2px dashed var(--border)',
            background: 'var(--bg-card)',
            cursor: uploading ? 'wait' : 'pointer',
            color: 'var(--text-secondary)',
            fontSize: '13px',
            transition: 'border-color 0.2s, background 0.2s',
            opacity: uploading ? 0.6 : 1,
          }}
          onMouseEnter={e => {
            if (!uploading) {
              e.currentTarget.style.borderColor = 'var(--accent)'
              e.currentTarget.style.background = 'var(--bg-card-hover)'
            }
          }}
          onMouseLeave={e => {
            e.currentTarget.style.borderColor = 'var(--border)'
            e.currentTarget.style.background = 'var(--bg-card)'
          }}
        >
          {uploading ? (
            <>
              <RefreshCw size={18} style={{ animation: 'spin 1s linear infinite' }} />
              Analyzing reference...
            </>
          ) : (
            <>
              <Upload size={18} />
              Upload a reference image for analysis
            </>
          )}
        </label>

        {uploadError && (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            marginTop: '10px',
            padding: '10px 14px',
            borderRadius: '10px',
            background: 'var(--error)' + '15',
            color: 'var(--error)',
            fontSize: '12px',
          }}>
            <AlertCircle size={14} />
            {uploadError}
          </div>
        )}

        {uploadResult && <ReferenceResult analysis={uploadResult} />}
      </Section>

      {/* Empty State */}
      {colors.length === 0 && fonts.length === 0 && styleKeywords.length === 0 && voiceTraits.length === 0 && (
        <div style={{
          textAlign: 'center',
          padding: '32px 16px',
          color: 'var(--text-muted)',
          fontSize: '13px',
        }}>
          No brand data loaded. Configure your brand guidelines to see colors, fonts, and style here.
        </div>
      )}

      <style>{`@keyframes spin { from { transform: rotate(0deg) } to { transform: rotate(360deg) } }`}</style>
    </div>
  )
}
