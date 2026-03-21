import { useState, useEffect } from 'react'
import { Play, Square, ChevronDown, ChevronUp, Layout, Check, Loader, AlertCircle, Code } from 'lucide-react'

const LAYOUT_PRESETS = [
  { id: '16:9', label: '16:9', desc: 'Landscape' },
  { id: '9:16', label: '9:16', desc: 'Portrait' },
  { id: '1:1', label: '1:1', desc: 'Square' },
]

function FieldLabel({ children }) {
  return (
    <label style={{
      display: 'block',
      fontSize: '11px',
      fontWeight: 600,
      color: 'var(--text-muted)',
      textTransform: 'uppercase',
      letterSpacing: '0.5px',
      marginBottom: '6px',
    }}>
      {children}
    </label>
  )
}

function TextInput({ label, value, onChange, placeholder, multiline = false }) {
  const sharedStyle = {
    width: '100%',
    padding: '10px 12px',
    borderRadius: '10px',
    background: 'var(--bg-card)',
    border: '1px solid var(--border)',
    color: 'var(--text-primary)',
    fontSize: '13px',
    outline: 'none',
    fontFamily: 'inherit',
    transition: 'border-color 0.15s',
    resize: multiline ? 'vertical' : 'none',
    lineHeight: 1.4,
    boxSizing: 'border-box',
  }

  return (
    <div style={{ marginBottom: '14px' }}>
      <FieldLabel>{label}</FieldLabel>
      {multiline ? (
        <textarea
          value={value || ''}
          onChange={e => onChange(e.target.value)}
          placeholder={placeholder}
          rows={3}
          style={{ ...sharedStyle, minHeight: '72px' }}
          onFocus={e => e.currentTarget.style.borderColor = 'var(--accent)'}
          onBlur={e => e.currentTarget.style.borderColor = 'var(--border)'}
        />
      ) : (
        <input
          type="text"
          value={value || ''}
          onChange={e => onChange(e.target.value)}
          placeholder={placeholder}
          style={{ ...sharedStyle, minHeight: '44px' }}
          onFocus={e => e.currentTarget.style.borderColor = 'var(--accent)'}
          onBlur={e => e.currentTarget.style.borderColor = 'var(--border)'}
        />
      )}
    </div>
  )
}

function ContentTypeSelector({ contentTypes, selectedType, onSelect }) {
  if (!contentTypes || contentTypes.length === 0) return null

  return (
    <div style={{ marginBottom: '14px' }}>
      <FieldLabel>Content Type</FieldLabel>
      <div style={{
        display: 'flex',
        gap: '8px',
        overflowX: 'auto',
        paddingBottom: '4px',
        WebkitOverflowScrolling: 'touch',
      }}>
        {contentTypes.map(ct => {
          const id = typeof ct === 'string' ? ct : ct.id || ct.name
          const label = typeof ct === 'string' ? ct : ct.label || ct.name || ct.id
          const isSelected = selectedType === id
          return (
            <button
              key={id}
              onClick={() => onSelect(id)}
              style={{
                padding: '10px 16px',
                borderRadius: '10px',
                background: isSelected ? 'var(--accent)' : 'var(--bg-card)',
                border: isSelected ? '1px solid var(--accent)' : '1px solid var(--border)',
                color: isSelected ? '#fff' : 'var(--text-secondary)',
                fontSize: '12px',
                fontWeight: isSelected ? 600 : 400,
                cursor: 'pointer',
                whiteSpace: 'nowrap',
                flexShrink: 0,
                transition: 'all 0.15s',
                minHeight: '44px',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
              }}
              onMouseEnter={e => {
                if (!isSelected) e.currentTarget.style.background = 'var(--bg-card-hover)'
              }}
              onMouseLeave={e => {
                if (!isSelected) e.currentTarget.style.background = 'var(--bg-card)'
              }}
            >
              {isSelected && <Check size={12} />}
              {label}
            </button>
          )
        })}
      </div>
    </div>
  )
}

function LayoutSelector({ selected, onSelect }) {
  return (
    <div style={{ marginBottom: '14px' }}>
      <FieldLabel>Layout</FieldLabel>
      <div style={{ display: 'flex', gap: '8px' }}>
        {LAYOUT_PRESETS.map(preset => {
          const isSelected = selected === preset.id
          // Compute a visual aspect ratio box
          const boxW = preset.id === '9:16' ? 28 : preset.id === '1:1' ? 36 : 48
          const boxH = preset.id === '9:16' ? 48 : preset.id === '1:1' ? 36 : 28
          return (
            <button
              key={preset.id}
              onClick={() => onSelect(preset.id)}
              style={{
                flex: 1,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: '8px',
                padding: '14px 8px',
                borderRadius: '12px',
                background: isSelected ? 'var(--accent-glow)' : 'var(--bg-card)',
                border: isSelected ? '2px solid var(--accent)' : '1px solid var(--border)',
                color: isSelected ? 'var(--accent)' : 'var(--text-secondary)',
                cursor: 'pointer',
                transition: 'all 0.15s',
                minHeight: '44px',
              }}
              onMouseEnter={e => {
                if (!isSelected) e.currentTarget.style.background = 'var(--bg-card-hover)'
              }}
              onMouseLeave={e => {
                if (!isSelected) e.currentTarget.style.background = isSelected ? 'var(--accent-glow)' : 'var(--bg-card)'
              }}
            >
              <div style={{
                width: `${boxW}px`,
                height: `${boxH}px`,
                borderRadius: '4px',
                border: `2px solid ${isSelected ? 'var(--accent)' : 'var(--border)'}`,
                transition: 'border-color 0.15s',
              }} />
              <div>
                <div style={{ fontSize: '12px', fontWeight: 600 }}>{preset.label}</div>
                <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>{preset.desc}</div>
              </div>
            </button>
          )
        })}
      </div>
    </div>
  )
}

function TemplatePreview({ template, templates }) {
  if (!template) return null
  const tpl = templates?.find(t => t.id === template)
  if (!tpl) {
    return (
      <div style={{ marginBottom: '14px' }}>
        <FieldLabel>Template</FieldLabel>
        <div style={{
          padding: '10px 14px',
          borderRadius: '10px',
          background: 'var(--bg-card)',
          border: '1px solid var(--border)',
          fontSize: '12px',
          fontFamily: 'var(--mono)',
          color: 'var(--text-secondary)',
        }}>
          {template}
        </div>
      </div>
    )
  }
  return (
    <div style={{ marginBottom: '14px' }}>
      <FieldLabel>Template</FieldLabel>
      <div style={{
        padding: '12px 14px',
        borderRadius: '10px',
        background: 'var(--accent-glow)',
        border: '1px solid var(--accent)',
        display: 'flex',
        alignItems: 'center',
        gap: '10px',
      }}>
        <Layout size={16} style={{ color: 'var(--accent)' }} />
        <div>
          <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>
            {tpl.name || tpl.id}
          </div>
          {tpl.aspect_ratio && (
            <span style={{ fontSize: '10px', color: 'var(--text-muted)', fontFamily: 'var(--mono)' }}>
              {tpl.aspect_ratio}
            </span>
          )}
        </div>
      </div>
    </div>
  )
}

function ProgressView({ status }) {
  return (
    <div style={{
      position: 'fixed',
      inset: 0,
      background: 'rgba(0,0,0,0.7)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 100,
      padding: '24px',
    }}>
      <div style={{
        background: 'var(--bg-secondary)',
        border: '1px solid var(--border)',
        borderRadius: '16px',
        padding: '32px 24px',
        width: '100%',
        maxWidth: '320px',
        textAlign: 'center',
      }}>
        <Loader size={32} style={{ color: 'var(--accent)', animation: 'spin 1s linear infinite', marginBottom: '16px' }} />
        <div style={{ fontSize: '15px', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '8px' }}>
          Generating
        </div>
        <div style={{ fontSize: '12px', color: 'var(--text-muted)', lineHeight: 1.5 }}>
          {status || 'Building your design...'}
        </div>
        <style>{`@keyframes spin { from { transform: rotate(0deg) } to { transform: rotate(360deg) } }`}</style>
      </div>
    </div>
  )
}

function GenerationResult({ result, onDismiss }) {
  return (
    <div style={{
      position: 'fixed',
      inset: 0,
      background: 'rgba(0,0,0,0.7)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 100,
      padding: '24px',
    }}>
      <div style={{
        background: 'var(--bg-secondary)',
        border: '1px solid var(--border)',
        borderRadius: '16px',
        padding: '24px',
        width: '100%',
        maxWidth: '360px',
        maxHeight: '80vh',
        overflow: 'auto',
      }}>
        {result.image_url && (
          <img
            src={result.image_url}
            alt="Generated design"
            style={{
              width: '100%',
              borderRadius: '12px',
              marginBottom: '16px',
            }}
          />
        )}
        {result.caption && (
          <p style={{
            fontSize: '13px',
            color: 'var(--text-primary)',
            lineHeight: 1.5,
            marginBottom: '12px',
            whiteSpace: 'pre-wrap',
          }}>
            {result.caption}
          </p>
        )}
        {result.error && (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '10px 14px',
            borderRadius: '10px',
            background: 'var(--error)15',
            color: 'var(--error)',
            fontSize: '12px',
            marginBottom: '12px',
          }}>
            <AlertCircle size={14} />
            {result.error}
          </div>
        )}
        <button
          onClick={onDismiss}
          style={{
            width: '100%',
            padding: '12px',
            borderRadius: '12px',
            background: 'var(--bg-card)',
            border: '1px solid var(--border)',
            color: 'var(--text-primary)',
            fontSize: '13px',
            fontWeight: 600,
            cursor: 'pointer',
            minHeight: '44px',
            transition: 'background 0.15s',
          }}
          onMouseEnter={e => e.currentTarget.style.background = 'var(--bg-card-hover)'}
          onMouseLeave={e => e.currentTarget.style.background = 'var(--bg-card)'}
        >
          Done
        </button>
      </div>
    </div>
  )
}

export default function Composer({
  designSpec,
  setDesignSpec,
  brandData,
  templates,
  contentTypes,
  sessionUploads,
  api,
}) {
  const [showSpec, setShowSpec] = useState(false)
  const [generating, setGenerating] = useState(false)
  const [genStatus, setGenStatus] = useState('')
  const [result, setResult] = useState(null)

  const update = (key, value) => {
    setDesignSpec(prev => ({ ...prev, [key]: value }))
  }

  const handleGenerate = async () => {
    setGenerating(true)
    setGenStatus('Preparing design spec...')
    setResult(null)

    // Build the spec — only include fields the backend DesignSpec model accepts
    const body = {}
    if (designSpec.content_type) body.content_type = designSpec.content_type
    if (designSpec.template_id) body.template_id = designSpec.template_id
    if (designSpec.title) body.title = designSpec.title
    if (designSpec.subtitle) body.subtitle = designSpec.subtitle
    if (designSpec.caption_guidance) body.caption_guidance = designSpec.caption_guidance
    if (designSpec.image_prompt) body.image_prompt = designSpec.image_prompt
    if (designSpec.color_overrides) body.color_overrides = designSpec.color_overrides
    if (designSpec.layout_preset) body.layout_preset = designSpec.layout_preset
    if (designSpec.style_notes) body.style_notes = designSpec.style_notes

    try {
      const headers = { 'Content-Type': 'application/json' }
      const initData = window.Telegram?.WebApp?.initData
      if (initData) headers['X-Telegram-InitData'] = initData

      const res = await fetch('/api/design/generate', {
        method: 'POST',
        headers,
        body: JSON.stringify(body),
      })

      const contentType = res.headers.get('content-type') || ''

      if (contentType.includes('text/event-stream')) {
        // SSE streaming
        const reader = res.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ''

        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split('\n')
          buffer = lines.pop() || ''
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const evt = JSON.parse(line.slice(6))
                if (evt.type === 'progress') {
                  setGenStatus(evt.message || evt.step || 'Working...')
                } else if (evt.type === 'result') {
                  setResult({
                    image_url: evt.image_url,
                    caption: evt.draft?.caption,
                    draft: evt.draft,
                    text: evt.text,
                  })
                } else if (evt.type === 'error') {
                  setResult({ error: evt.message })
                }
              } catch {
                // Not JSON, ignore
              }
            }
          }
        }
      } else {
        // Regular JSON response
        if (!res.ok) throw new Error(`API error: ${res.status}`)
        const data = await res.json()
        setResult(data)
      }
    } catch (e) {
      setResult({ error: e.message || 'Generation failed' })
    }

    setGenerating(false)
    setGenStatus('')
  }

  const canGenerate = designSpec.content_type || designSpec.title || designSpec.image_prompt

  return (
    <div style={{ padding: '16px', paddingBottom: '80px' }}>
      {/* Content Type */}
      <ContentTypeSelector
        contentTypes={contentTypes}
        selectedType={designSpec.content_type}
        onSelect={v => update('content_type', v)}
      />

      {/* Template */}
      <TemplatePreview
        template={designSpec.template_id}
        templates={templates}
      />

      {/* Layout */}
      <LayoutSelector
        selected={designSpec.layout_preset || '1:1'}
        onSelect={v => update('layout_preset', v)}
      />

      {/* Text Fields */}
      <TextInput
        label="Title"
        value={designSpec.title}
        onChange={v => update('title', v)}
        placeholder="Main heading text"
      />

      <TextInput
        label="Subtitle"
        value={designSpec.subtitle}
        onChange={v => update('subtitle', v)}
        placeholder="Supporting text"
      />

      <TextInput
        label="Caption Guidance"
        value={designSpec.caption_guidance}
        onChange={v => update('caption_guidance', v)}
        placeholder="How should the caption read?"
        multiline
      />

      <TextInput
        label="Image Prompt"
        value={designSpec.image_prompt}
        onChange={v => update('image_prompt', v)}
        placeholder="Describe the visual style and composition"
        multiline
      />

      <TextInput
        label="Style Notes"
        value={designSpec.style_notes}
        onChange={v => update('style_notes', v)}
        placeholder="Additional style directions"
        multiline
      />

      {/* JSON Spec Toggle */}
      <button
        onClick={() => setShowSpec(!showSpec)}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          padding: '8px 0',
          background: 'none',
          border: 'none',
          color: 'var(--text-muted)',
          fontSize: '11px',
          fontFamily: 'var(--mono)',
          cursor: 'pointer',
          marginBottom: '8px',
        }}
      >
        <Code size={12} />
        {showSpec ? 'Hide' : 'Show'} spec JSON
        {showSpec ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
      </button>

      {showSpec && (
        <pre style={{
          padding: '12px',
          borderRadius: '10px',
          background: 'var(--bg-card)',
          border: '1px solid var(--border)',
          color: 'var(--text-secondary)',
          fontSize: '11px',
          fontFamily: 'var(--mono)',
          overflow: 'auto',
          maxHeight: '200px',
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
          marginBottom: '14px',
          lineHeight: 1.5,
        }}>
          {JSON.stringify(designSpec, null, 2)}
        </pre>
      )}

      {/* Generate Button */}
      <button
        onClick={handleGenerate}
        disabled={!canGenerate || generating}
        style={{
          position: 'fixed',
          bottom: '70px',
          left: '16px',
          right: '16px',
          padding: '14px',
          borderRadius: '14px',
          background: canGenerate && !generating ? 'var(--accent)' : 'var(--bg-card)',
          border: canGenerate && !generating ? 'none' : '1px solid var(--border)',
          color: canGenerate && !generating ? '#fff' : 'var(--text-muted)',
          fontSize: '15px',
          fontWeight: 700,
          cursor: canGenerate && !generating ? 'pointer' : 'default',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '8px',
          transition: 'all 0.15s',
          zIndex: 50,
          minHeight: '50px',
          boxShadow: canGenerate ? '0 4px 20px rgba(59,130,246,0.3)' : 'none',
        }}
        onMouseEnter={e => {
          if (canGenerate && !generating) e.currentTarget.style.opacity = '0.9'
        }}
        onMouseLeave={e => e.currentTarget.style.opacity = '1'}
      >
        {generating ? (
          <>
            <Square size={16} />
            Generating...
          </>
        ) : (
          <>
            <Play size={16} />
            Generate
          </>
        )}
      </button>

      {/* Progress Overlay */}
      {generating && <ProgressView status={genStatus} />}

      {/* Result Overlay */}
      {result && !generating && (
        <GenerationResult
          result={result}
          onDismiss={() => setResult(null)}
        />
      )}
    </div>
  )
}
