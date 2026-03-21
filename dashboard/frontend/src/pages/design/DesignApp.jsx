import { useState, useEffect } from 'react'
import { Palette, LayoutTemplate, PenTool, MessageSquare, History } from 'lucide-react'
import { useTelegram } from '../../hooks/useTelegram'
import { useDesignApi } from '../../hooks/useDesignApi'
import BrandBoard from './BrandBoard'
import TemplateGallery from './TemplateGallery'
import Composer from './Composer'
import DesignChat from './DesignChat'
import HistoryView from './HistoryView'

const TABS = [
  { id: 'board', label: 'Brand', icon: Palette },
  { id: 'templates', label: 'Templates', icon: LayoutTemplate },
  { id: 'chat', label: 'Agent', icon: MessageSquare },
  { id: 'compose', label: 'Compose', icon: PenTool },
  { id: 'history', label: 'History', icon: History },
]

export default function DesignApp() {
  const [activeTab, setActiveTab] = useState('board')
  const [brandData, setBrandData] = useState(null)
  const [templates, setTemplates] = useState([])
  const [contentTypes, setContentTypes] = useState([])
  const [designSpec, setDesignSpec] = useState({})
  const [sessionUploads, setSessionUploads] = useState([])
  const { isTelegram, haptic } = useTelegram()
  const api = useDesignApi()

  // Load brand data on mount
  useEffect(() => {
    async function load() {
      const [board, tpls, cts] = await Promise.all([
        api.get('/brand-board'),
        api.get('/templates'),
        api.get('/content-types'),
      ])
      if (board) setBrandData(board)
      if (tpls) setTemplates(tpls.templates || [])
      if (cts) setContentTypes(cts.content_types || [])
    }
    load()
  }, [])

  const handleTabChange = (tabId) => {
    setActiveTab(tabId)
    haptic.selection()
  }

  // When the Design Agent produces a spec, pre-fill the composer
  const handleSpecFromAgent = (spec) => {
    setDesignSpec(spec)
    setActiveTab('compose')
    haptic.notification('success')
  }

  // When user selects a template, move to compose
  const handleTemplateSelect = (template) => {
    setDesignSpec(prev => ({ ...prev, template_id: template.id }))
    setActiveTab('compose')
    haptic.selection()
  }

  // When reference is analyzed, add to session
  const handleReferenceAnalyzed = (analysis) => {
    setSessionUploads(prev => [...prev, analysis])
    haptic.notification('success')
  }

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100vh',
      background: 'var(--bg-primary)',
      color: 'var(--text-primary)',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    }}>
      {/* Header */}
      <div style={{
        padding: '12px 16px',
        background: 'var(--bg-secondary)',
        borderBottom: '1px solid var(--border)',
        display: 'flex',
        alignItems: 'center',
        gap: '10px',
        flexShrink: 0,
      }}>
        <PenTool size={20} style={{ color: 'var(--accent)' }} />
        <span style={{ fontWeight: 600, fontSize: '16px' }}>Design Studio</span>
        {brandData && (
          <span style={{
            fontSize: '12px',
            color: 'var(--text-muted)',
            marginLeft: 'auto',
          }}>
            {brandData.brand_name}
          </span>
        )}
      </div>

      {/* Tab Content */}
      <div style={{ flex: 1, overflow: 'auto', padding: '0' }}>
        {activeTab === 'board' && (
          <BrandBoard
            brandData={brandData}
            onReferenceAnalyzed={handleReferenceAnalyzed}
            api={api}
          />
        )}
        {activeTab === 'templates' && (
          <TemplateGallery
            templates={templates}
            onSelect={handleTemplateSelect}
          />
        )}
        {activeTab === 'chat' && (
          <DesignChat
            brandData={brandData}
            sessionUploads={sessionUploads}
            onSpecReady={handleSpecFromAgent}
            api={api}
          />
        )}
        {activeTab === 'compose' && (
          <Composer
            designSpec={designSpec}
            setDesignSpec={setDesignSpec}
            brandData={brandData}
            templates={templates}
            contentTypes={contentTypes}
            sessionUploads={sessionUploads}
            api={api}
          />
        )}
        {activeTab === 'history' && (
          <HistoryView api={api} />
        )}
      </div>

      {/* Bottom Tab Bar */}
      <div style={{
        display: 'flex',
        background: 'var(--bg-secondary)',
        borderTop: '1px solid var(--border)',
        padding: `8px 0 ${isTelegram ? 'env(safe-area-inset-bottom, 8px)' : '8px'}`,
        flexShrink: 0,
      }}>
        {TABS.map(tab => {
          const Icon = tab.icon
          const isActive = activeTab === tab.id
          return (
            <button
              key={tab.id}
              onClick={() => handleTabChange(tab.id)}
              style={{
                flex: 1,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: '4px',
                padding: '6px 0',
                background: 'none',
                border: 'none',
                color: isActive ? 'var(--accent)' : 'var(--text-muted)',
                cursor: 'pointer',
                fontSize: '10px',
                fontWeight: isActive ? 600 : 400,
                transition: 'color 0.2s',
                minHeight: '44px',
                justifyContent: 'center',
              }}
            >
              <Icon size={20} />
              {tab.label}
            </button>
          )
        })}
      </div>
    </div>
  )
}
