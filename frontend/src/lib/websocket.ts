type Listener<T> = (data: T) => void

interface SocketOptions {
  path: string
  onStatusChange?: (status: 'connected' | 'connecting' | 'disconnected') => void
  maxBackoffMs?: number
}

/**
 * Minimal reconnecting WebSocket wrapper. Exponential backoff, no external
 * deps. Matches the backend's oms_signals-style event stream: every message
 * is JSON of shape { type: string, payload: unknown }.
 */
export class LiveSocket {
  private ws: WebSocket | null = null
  private listeners = new Map<string, Set<Listener<any>>>()
  private attempt = 0
  private closedByUser = false
  private readonly url: string
  private readonly opts: SocketOptions

  constructor(opts: SocketOptions) {
    this.opts = opts
    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const base = import.meta.env.VITE_WS_BASE_URL || `${proto}://${window.location.host}`
    this.url = `${base}${opts.path}`
  }

  connect() {
    this.closedByUser = false
    this.opts.onStatusChange?.('connecting')
    try {
      this.ws = new WebSocket(this.url)
    } catch {
      this.scheduleReconnect()
      return
    }

    this.ws.onopen = () => {
      this.attempt = 0
      this.opts.onStatusChange?.('connected')
    }

    this.ws.onmessage = (event) => {
      try {
        const { type, payload } = JSON.parse(event.data)
        this.listeners.get(type)?.forEach((fn) => fn(payload))
      } catch {
        // Malformed frame — drop it, don't crash the socket.
      }
    }

    this.ws.onclose = () => {
      this.opts.onStatusChange?.('disconnected')
      if (!this.closedByUser) this.scheduleReconnect()
    }

    this.ws.onerror = () => {
      this.ws?.close()
    }
  }

  private scheduleReconnect() {
    const max = this.opts.maxBackoffMs ?? 15_000
    const delay = Math.min(1000 * 2 ** this.attempt, max)
    this.attempt += 1
    setTimeout(() => {
      if (!this.closedByUser) this.connect()
    }, delay)
  }

  on<T = unknown>(type: string, fn: Listener<T>) {
    if (!this.listeners.has(type)) this.listeners.set(type, new Set())
    this.listeners.get(type)!.add(fn)
    return () => this.listeners.get(type)?.delete(fn)
  }

  close() {
    this.closedByUser = true
    this.ws?.close()
  }
}
