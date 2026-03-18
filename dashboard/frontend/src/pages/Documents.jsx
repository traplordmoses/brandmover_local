import { useState, useEffect } from 'react'
import { FileText, Save, Eye, Code, Circle, AlertCircle } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import useApi from '../hooks/useApi'

export default function DocumentsPage() {
  const { data: docs, loading, reload } = useApi('/api/documents')
  const [selectedFile, setSelectedFile] = useState(null)
  const [content, setContent] = useState('')
  const [originalContent, setOriginalContent] = useState('')
  const [viewMode, setViewMode] = useState('rendered')
  const [saving, setSaving] = useState(false)
  const [saveStatus, setSaveStatus] = useState(null)

  useEffect(() => {
    if (docs?.files?.length > 0 && !selectedFile) {
      setSelectedFile(docs.files[0].path)
    }
  }, [docs, selectedFile])

  useEffect(() => {
    if (selectedFile) {
      fetch(`/api/documents/content?path=${encodeURIComponent(selectedFile)}`)
        .then(r => r.json())
        .then(d => {
          setContent(d.content || '')
          setOriginalContent(d.content || '')
          setSaveStatus(null)
        })
        .catch(() => {
          setContent('Failed to load document')
          setOriginalContent('')
        })
    }
  }, [selectedFile])

  const hasUnsavedChanges = content !== originalContent

  const handleSave = async () => {
    setSaving(true)
    setSaveStatus(null)
    try {
      const r = await fetch('/api/documents/content', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: selectedFile, content }),
      })
      if (r.ok) {
        setOriginalContent(content)
        setSaveStatus('saved')
        setTimeout(() => setSaveStatus(null), 3000)
      } else {
        setSaveStatus('error')
      }
    } catch {
      setSaveStatus('error')
    }
    setSaving(false)
  }

  return (
    <div className="flex gap-4 h-[calc(100vh-7rem)]">
      {/* File sidebar */}
      <div className="w-52 flex-shrink-0 rounded-lg border overflow-auto" style={{ borderColor: 'var(--border)', background: 'var(--bg-card)' }}>
        <div className="px-3 py-2 border-b font-mono text-xs" style={{ borderColor: 'var(--border)', color: 'var(--text-muted)' }}>
          Brand Documents
        </div>
        {loading && (
          <div className="px-3 py-4 text-xs font-mono" style={{ color: 'var(--text-muted)' }}>Loading...</div>
        )}
        {docs?.files?.map(file => (
          <button
            key={file.path}
            onClick={() => setSelectedFile(file.path)}
            className="w-full text-left px-3 py-2 text-sm flex items-center gap-2 cursor-pointer transition-colors border-none"
            style={{
              background: selectedFile === file.path ? 'var(--accent-glow)' : 'transparent',
              color: selectedFile === file.path ? 'var(--accent)' : 'var(--text-secondary)',
            }}
          >
            <FileText size={14} />
            <span className="truncate">{file.name}</span>
          </button>
        ))}
        {docs?.files?.length === 0 && (
          <div className="px-3 py-4 text-xs font-mono" style={{ color: 'var(--text-muted)' }}>No documents found</div>
        )}
      </div>

      {/* Content area */}
      <div className="flex-1 rounded-lg border overflow-hidden flex flex-col" style={{ borderColor: 'var(--border)', background: 'var(--bg-card)' }}>
        {/* Toolbar */}
        <div className="flex items-center justify-between px-4 py-2 border-b" style={{ borderColor: 'var(--border)', background: 'var(--bg-secondary)' }}>
          <div className="flex items-center gap-3">
            <span className="font-mono text-xs" style={{ color: 'var(--text-muted)' }}>
              {selectedFile || 'No file selected'}
            </span>
            {hasUnsavedChanges && (
              <span className="flex items-center gap-1 text-xs font-mono" style={{ color: 'var(--warning)' }}>
                <Circle size={6} fill="var(--warning)" stroke="var(--warning)" />
                unsaved
              </span>
            )}
            {saveStatus === 'saved' && (
              <span className="text-xs font-mono" style={{ color: 'var(--success)' }}>saved</span>
            )}
            {saveStatus === 'error' && (
              <span className="flex items-center gap-1 text-xs font-mono" style={{ color: 'var(--error)' }}>
                <AlertCircle size={10} /> save failed
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            <div className="flex gap-1">
              <button
                onClick={() => setViewMode('rendered')}
                className="flex items-center gap-1 px-2 py-1 rounded text-xs cursor-pointer transition-colors"
                style={{
                  background: viewMode === 'rendered' ? 'var(--accent-glow)' : 'transparent',
                  color: viewMode === 'rendered' ? 'var(--accent)' : 'var(--text-muted)',
                  border: 'none',
                }}
              >
                <Eye size={12} /> Preview
              </button>
              <button
                onClick={() => setViewMode('raw')}
                className="flex items-center gap-1 px-2 py-1 rounded text-xs cursor-pointer transition-colors"
                style={{
                  background: viewMode === 'raw' ? 'var(--accent-glow)' : 'transparent',
                  color: viewMode === 'raw' ? 'var(--accent)' : 'var(--text-muted)',
                  border: 'none',
                }}
              >
                <Code size={12} /> Edit
              </button>
            </div>
            <button
              onClick={handleSave}
              disabled={!hasUnsavedChanges || saving}
              className="flex items-center gap-1 px-3 py-1 rounded-lg border text-xs font-mono cursor-pointer transition-opacity"
              style={{
                borderColor: hasUnsavedChanges ? 'var(--accent)' : 'var(--border)',
                color: hasUnsavedChanges ? 'var(--accent)' : 'var(--text-muted)',
                background: 'transparent',
                opacity: hasUnsavedChanges ? 1 : 0.5,
              }}
            >
              <Save size={12} /> {saving ? 'Saving...' : 'Save'}
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto p-4">
          {!selectedFile && (
            <div className="flex items-center justify-center h-full font-mono text-xs" style={{ color: 'var(--text-muted)' }}>
              Select a document
            </div>
          )}
          {selectedFile && viewMode === 'rendered' && (
            <div className="prose prose-invert max-w-none text-sm" style={{ color: 'var(--text-secondary)' }}>
              <ReactMarkdown
                components={{
                  h1: ({ children }) => <h1 style={{ color: 'var(--text-primary)', fontSize: '1.5rem', fontWeight: 700, marginBottom: '0.5rem', marginTop: '1.5rem' }}>{children}</h1>,
                  h2: ({ children }) => <h2 style={{ color: 'var(--text-primary)', fontSize: '1.25rem', fontWeight: 600, marginBottom: '0.5rem', marginTop: '1.25rem' }}>{children}</h2>,
                  h3: ({ children }) => <h3 style={{ color: 'var(--text-primary)', fontSize: '1.1rem', fontWeight: 600, marginBottom: '0.25rem', marginTop: '1rem' }}>{children}</h3>,
                  p: ({ children }) => <p style={{ color: 'var(--text-secondary)', marginBottom: '0.75rem', lineHeight: 1.6 }}>{children}</p>,
                  ul: ({ children }) => <ul style={{ paddingLeft: '1.5rem', marginBottom: '0.75rem' }}>{children}</ul>,
                  ol: ({ children }) => <ol style={{ paddingLeft: '1.5rem', marginBottom: '0.75rem' }}>{children}</ol>,
                  li: ({ children }) => <li style={{ color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>{children}</li>,
                  code: ({ children, className }) => {
                    const isBlock = className?.includes('language-')
                    if (isBlock) {
                      return (
                        <pre style={{ background: 'var(--bg-secondary)', padding: '1rem', borderRadius: '0.5rem', overflow: 'auto', marginBottom: '0.75rem' }}>
                          <code style={{ fontFamily: 'var(--mono)', fontSize: '0.8rem', color: 'var(--text-primary)' }}>{children}</code>
                        </pre>
                      )
                    }
                    return <code style={{ fontFamily: 'var(--mono)', fontSize: '0.85rem', background: 'var(--bg-secondary)', padding: '0.125rem 0.375rem', borderRadius: '0.25rem', color: 'var(--accent)' }}>{children}</code>
                  },
                  table: ({ children }) => (
                    <div style={{ overflow: 'auto', marginBottom: '0.75rem' }}>
                      <table style={{ borderCollapse: 'collapse', width: '100%', fontSize: '0.85rem' }}>{children}</table>
                    </div>
                  ),
                  th: ({ children }) => <th style={{ textAlign: 'left', padding: '0.5rem', borderBottom: '1px solid var(--border)', color: 'var(--text-primary)', fontWeight: 600 }}>{children}</th>,
                  td: ({ children }) => <td style={{ padding: '0.5rem', borderBottom: '1px solid var(--border)', color: 'var(--text-secondary)' }}>{children}</td>,
                  blockquote: ({ children }) => <blockquote style={{ borderLeft: '3px solid var(--accent)', paddingLeft: '1rem', margin: '0.75rem 0', color: 'var(--text-muted)' }}>{children}</blockquote>,
                  hr: () => <hr style={{ border: 'none', borderTop: '1px solid var(--border)', margin: '1.5rem 0' }} />,
                  a: ({ children, href }) => <a href={href} target="_blank" rel="noopener noreferrer" style={{ color: 'var(--accent)', textDecoration: 'none' }}>{children}</a>,
                }}
              >
                {content}
              </ReactMarkdown>
            </div>
          )}
          {selectedFile && viewMode === 'raw' && (
            <textarea
              value={content}
              onChange={e => setContent(e.target.value)}
              className="w-full h-full resize-none border-none outline-none text-sm"
              style={{
                background: 'transparent',
                color: 'var(--text-primary)',
                fontFamily: 'var(--mono)',
                lineHeight: 1.6,
              }}
              spellCheck={false}
            />
          )}
        </div>
      </div>
    </div>
  )
}
