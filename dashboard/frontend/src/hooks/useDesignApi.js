import { useState, useCallback } from 'react'

const API_BASE = '/api/design'

export function useDesignApi() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const headers = () => {
    const h = { 'Content-Type': 'application/json' }
    // Add Telegram initData if available
    const initData = window.Telegram?.WebApp?.initData
    if (initData) h['X-Telegram-InitData'] = initData
    return h
  }

  const get = useCallback(async (path) => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`${API_BASE}${path}`, { headers: headers() })
      if (!res.ok) throw new Error(`API error: ${res.status}`)
      return await res.json()
    } catch (e) {
      setError(e.message)
      return null
    } finally {
      setLoading(false)
    }
  }, [])

  const post = useCallback(async (path, body) => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`${API_BASE}${path}`, {
        method: 'POST',
        headers: headers(),
        body: JSON.stringify(body),
      })
      if (!res.ok) throw new Error(`API error: ${res.status}`)
      return await res.json()
    } catch (e) {
      setError(e.message)
      return null
    } finally {
      setLoading(false)
    }
  }, [])

  const upload = useCallback(async (path, file) => {
    setLoading(true)
    setError(null)
    try {
      const form = new FormData()
      form.append('file', file)
      const h = {}
      const initData = window.Telegram?.WebApp?.initData
      if (initData) h['X-Telegram-InitData'] = initData
      const res = await fetch(`${API_BASE}${path}`, {
        method: 'POST',
        headers: h,
        body: form,
      })
      if (!res.ok) throw new Error(`API error: ${res.status}`)
      return await res.json()
    } catch (e) {
      setError(e.message)
      return null
    } finally {
      setLoading(false)
    }
  }, [])

  return { get, post, upload, loading, error }
}
