import { useEffect, useState } from 'react'

export function useTelegram() {
  const [webApp, setWebApp] = useState(null)
  const [user, setUser] = useState(null)
  const [themeParams, setThemeParams] = useState({})
  const [isTelegram, setIsTelegram] = useState(false)

  useEffect(() => {
    const tg = window.Telegram?.WebApp
    if (tg) {
      tg.ready()
      tg.expand() // Full screen
      setWebApp(tg)
      setUser(tg.initDataUnsafe?.user || null)
      setThemeParams(tg.themeParams || {})
      setIsTelegram(true)

      // Apply Telegram theme colors as CSS variables
      if (tg.themeParams) {
        const root = document.documentElement
        if (tg.themeParams.bg_color) root.style.setProperty('--bg-primary', tg.themeParams.bg_color)
        if (tg.themeParams.secondary_bg_color) root.style.setProperty('--bg-secondary', tg.themeParams.secondary_bg_color)
        if (tg.themeParams.text_color) root.style.setProperty('--text-primary', tg.themeParams.text_color)
        if (tg.themeParams.hint_color) root.style.setProperty('--text-secondary', tg.themeParams.hint_color)
        if (tg.themeParams.button_color) root.style.setProperty('--accent', tg.themeParams.button_color)
      }
    }
  }, [])

  const initData = webApp?.initData || ''

  const haptic = {
    impact: (style = 'medium') => webApp?.HapticFeedback?.impactOccurred(style),
    notification: (type = 'success') => webApp?.HapticFeedback?.notificationOccurred(type),
    selection: () => webApp?.HapticFeedback?.selectionChanged(),
  }

  return { webApp, user, themeParams, isTelegram, initData, haptic }
}
