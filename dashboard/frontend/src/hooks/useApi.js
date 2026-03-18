import { useState, useEffect, useCallback } from 'react'

export default function useApi(url, interval = null) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const reload = useCallback(() => {
    setLoading(true)
    fetch(url)
      .then(r => {
        if (!r.ok) throw new Error(`${r.status} ${r.statusText}`)
        return r.json()
      })
      .then(d => { setData(d); setLoading(false); setError(null) })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [url])

  useEffect(() => {
    reload()
    if (interval) {
      const id = setInterval(reload, interval)
      return () => clearInterval(id)
    }
  }, [reload, interval])

  return { data, loading, error, reload }
}
