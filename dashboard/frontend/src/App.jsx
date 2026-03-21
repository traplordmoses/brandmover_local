import { BrowserRouter, Routes, Route, NavLink, useLocation } from 'react-router-dom'
import { Calendar, Activity, FileText, Megaphone, Settings, PenTool } from 'lucide-react'
import CalendarPage from './pages/Calendar'
import StatusPage from './pages/Status'
import DocumentsPage from './pages/Documents'
import CampaignsPage from './pages/Campaigns'
import SettingsPage from './pages/Settings'
import DesignApp from './pages/design/DesignApp'
import StatusBar from './components/StatusBar'

const NAV = [
  { to: '/', icon: Calendar, label: 'Calendar' },
  { to: '/status', icon: Activity, label: 'Status' },
  { to: '/documents', icon: FileText, label: 'Brand Docs' },
  { to: '/campaigns', icon: Megaphone, label: 'Campaigns' },
  { to: '/design', icon: PenTool, label: 'Design Studio' },
  { to: '/settings', icon: Settings, label: 'Settings' },
]

function AppShell() {
  const location = useLocation()
  const isTelegram = !!window.Telegram?.WebApp
  const isDesignRoute = location.pathname === '/design'

  // In Telegram Mini App mode, render DesignApp full-screen without sidebar
  if (isTelegram && isDesignRoute) {
    return <DesignApp />
  }

  return (
    <div className="min-h-screen flex flex-col" style={{ background: 'var(--bg-primary)' }}>
      <StatusBar />
      <div className="flex flex-1">
        <nav className="w-56 border-r flex-shrink-0 p-4 flex flex-col gap-1" style={{ borderColor: 'var(--border)', background: 'var(--bg-secondary)' }}>
          <div className="px-3 py-4 mb-4">
            <h1 className="text-lg font-bold tracking-tight" style={{ color: 'var(--text-primary)' }}>BrandMover</h1>
            <p className="text-xs mt-1 font-mono" style={{ color: 'var(--text-muted)' }}>mission control</p>
          </div>
          {NAV.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors ${
                  isActive ? 'font-medium' : ''
                }`
              }
              style={({ isActive }) => ({
                background: isActive ? 'var(--accent-glow)' : 'transparent',
                color: isActive ? 'var(--accent)' : 'var(--text-secondary)',
              })}
            >
              <Icon size={18} />
              {label}
            </NavLink>
          ))}
        </nav>
        <main className="flex-1 p-6 overflow-auto">
          <Routes>
            <Route path="/" element={<CalendarPage />} />
            <Route path="/status" element={<StatusPage />} />
            <Route path="/documents" element={<DocumentsPage />} />
            <Route path="/campaigns" element={<CampaignsPage />} />
            <Route path="/design" element={<DesignApp />} />
            <Route path="/settings" element={<SettingsPage />} />
          </Routes>
        </main>
      </div>
    </div>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <AppShell />
    </BrowserRouter>
  )
}
